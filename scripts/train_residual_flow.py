#!/usr/bin/env python3
"""Residual flow-matching head on top of a FROZEN MSE adapter.

Idea (from ~/investigation_summary_2026-07-17.md, Phase 4 synthesis):
  The MSE adapter (`DA3ToNOVA3RAlignment`) collapses the multimodal token target to its
  conditional mean; the final linear layer `out_proj` is what discards the mode structure
  ("the final linear layer was discarding it, not the backbone" -- penultimate features
  retain ~90% of the target mode spread). So:

    * START the flow at the MSE prediction z_pred = out_proj(norm(query_tokens))
      -> short, structured transport instead of the ~2.2 NN precision floor that
         noise-start sampling hit.
    * CONDITION each token's flow on that token's penultimate feature
      norm(query_tokens) [768 x 512], which carries the mode structure z_pred dropped.
      This is the ONLY thing that escapes the Phase-2 "information wall" (refining the
      regressor's output from the regressor's own scalar output learns nothing -- the
      optimum is zero velocity). The 512-d feature is a strict superset of the 128-d
      z_pred, so the head has genuinely more to work with.
    * PER-TOKEN, no Hungarian at inference (matching is only used once, at bank-build
      time, to define each prediction token's residual target).

  Straight conditional flow matching (rectified flow): z_t = (1-t) z_pred + t z_tgt,
  target velocity v* = z_tgt - z_pred (constant along the path). Inference integrates
  z_1 = z_pred + int_0^1 head(z_t, t, cond) dt with a few Euler/midpoint steps, then
  decode(z_1).

HONEST PRIOR: the summary reports this collapsed to regressor level when amortised over
2400 windows (only the one-window overfit succeeded, 0.30->0.66 F@2cm). So the decode-eval
here measures the real signal -- Chamfer(decode(z_flow), decode(z_tgt)) vs
Chamfer(decode(z_pred), decode(z_tgt)) -- on train AND val windows, and --one-window-sanity
reproduces the known-good case first to validate the plumbing before spending a full run.

Usage:
  # 1) sanity: overfit one window, decode gap to oracle must -> ~0
  python scripts/train_residual_flow.py --one-window-sanity \
      --base-ckpt outputs/consecutive_windows/fullrec_nf16_vel03_best.pt \
      --rooms scripts/data/fc_nf16_span24_100_l13/room0

  # 2) full multi-window run
  python scripts/train_residual_flow.py \
      --base-ckpt outputs/consecutive_windows/fullrec_nf16_vel03_best.pt \
      --rooms scripts/data/fc_nf16_span24_100_l13/office0 ... \
      --val-rooms scripts/data/fc_nf16_span24_100_l13/office4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "da3" / "src"),
           str(REPO_ROOT / "nova3r_lib" / "third_party"), str(REPO_ROOT / "nova3r_lib")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_online_var_hungarian import (          # noqa: E402
    parse_root_configs, discover_cached_windows, load_window, load_pts_norm,
    _batch_match_cols,
)
# Importing the trainer prepends the bare `nova3r/` copy (no scene_ae checkpoint, different
# decoder code) to sys.path. Re-assert nova3r_lib at the front so `demo_nova3r` and the
# `nova3r.*` package resolve to the active checkpointed copy, exactly like stitch_office4.
for _p in [str(REPO_ROOT / "nova3r_lib" / "third_party"), str(REPO_ROOT / "nova3r_lib")]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402

CFG_PATH = REPO_ROOT / "experiments" / "overfit_8frames" / "config.yaml"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ base adapter
def adapter_cfg_from_sd(sd: dict) -> dict:
    cfg = dict(
        source_dim=sd["source_proj.weight"].shape[1],
        hidden_dim=sd["source_proj.weight"].shape[0],
        target_tokens=sd["target_queries"].shape[0],
        target_dim=sd["out_proj.weight"].shape[0],
        depth=sum(1 for k in sd if k.endswith(".self_attn.in_proj_weight")),
        num_heads=8,
    )
    if "geom_embed.0.weight" in sd:
        cfg["geom_cond"] = True
        cfg["geom_bands"] = sd["geom_embed.0.weight"].shape[1] // 6
    return cfg


def load_frozen_adapter(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    acfg = adapter_cfg_from_sd(sd)
    adapter = DA3ToNOVA3RAlignment(**acfg, drop=0.0).to(DEV)
    adapter.load_state_dict(sd)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    print(f"[base] loaded frozen adapter {ckpt_path.name}  cfg={acfg}")
    return adapter, acfg


@torch.no_grad()
def adapter_pred_and_feats(adapter, da3_tokens):
    """Return z_pred [B,768,128] and penultimate feats [B,768,H] via a pre-hook on out_proj.

    The pre-hook captures out_proj's input, which is exactly norm(query_tokens) -- the
    mode-carrying penultimate feature the summary identifies.
    """
    feats = {}

    def _pre_hook(_module, inputs):
        feats["h"] = inputs[0].detach()

    handle = adapter.out_proj.register_forward_pre_hook(_pre_hook)
    try:
        z_pred = adapter(da3_tokens.to(DEV))
    finally:
        handle.remove()
    return z_pred, feats["h"]


# ------------------------------------------------------------------ bank build
def build_bank(adapter, win_dirs, tag: str):
    """One frozen forward per window -> (z_pred, cond_feats, z_tgt matched to z_pred)."""
    zp, cf, zt = [], [], []
    for i, wd in enumerate(win_dirs):
        _, da3, mean, _ = load_window(wd)
        z_pred, feats = adapter_pred_and_feats(adapter, da3.unsqueeze(0) if da3.ndim == 2 else da3)
        # Hungarian-match target columns to the prediction (once, at bank build).
        cols = _batch_match_cols(z_pred, mean.unsqueeze(0).to(DEV))[0]
        col = torch.as_tensor(cols, dtype=torch.long)
        zp.append(z_pred[0].cpu())
        cf.append(feats[0].cpu())
        zt.append(mean[col])
        if (i + 1) % 100 == 0 or i + 1 == len(win_dirs):
            print(f"[bank:{tag}] {i+1}/{len(win_dirs)} windows")
    bank = dict(z_pred=torch.stack(zp), cond=torch.stack(cf), z_tgt=torch.stack(zt),
                win_dirs=[str(w) for w in win_dirs])
    res = (bank["z_tgt"] - bank["z_pred"])
    print(f"[bank:{tag}] z_pred{tuple(bank['z_pred'].shape)} cond{tuple(bank['cond'].shape)}  "
          f"residual rms={res.pow(2).mean().sqrt():.4f}  |z_pred|rms={bank['z_pred'].pow(2).mean().sqrt():.4f}")
    return bank


# ------------------------------------------------------------------ flow head
class TokenResidualFlow(torch.nn.Module):
    """Per-token conditional velocity field v(z_t, t | cond). Tokens are independent;
    all 768 are processed in parallel by folding them into the batch dim."""

    def __init__(self, token_dim=128, cond_dim=512, hidden=512, depth=4, time_bands=64):
        super().__init__()
        self.time_bands = time_bands
        self.inp = torch.nn.Linear(token_dim + cond_dim + 2 * time_bands, hidden)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, hidden),
                                torch.nn.GELU(), torch.nn.Linear(hidden, hidden))
            for _ in range(depth)
        ])
        self.out = torch.nn.Linear(hidden, token_dim)
        self.out.weight.data.mul_(0.1)   # start near identity flow (small velocity)
        self.out.bias.data.zero_()

    def _time_emb(self, t):
        freqs = (2.0 ** torch.arange(self.time_bands, device=t.device, dtype=t.dtype)) * math.pi
        a = t[..., None] * freqs
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, z_t, t, cond):
        # z_t [B,N,128], cond [B,N,C], t [B,N]
        h = torch.cat([z_t, cond, self._time_emb(t)], dim=-1)
        h = self.inp(h)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def cfm_step(head, z_pred, z_tgt, cond, gen):
    """Straight (rectified) conditional flow-matching loss."""
    B, N, _ = z_pred.shape
    t = torch.rand(B, N, device=z_pred.device, generator=gen)
    z_t = (1 - t)[..., None] * z_pred + t[..., None] * z_tgt
    v_star = z_tgt - z_pred                     # constant velocity of the straight path
    v_pred = head(z_t, t, cond)
    return (v_pred - v_star).square().mean()


@torch.no_grad()
def integrate(head, z_pred, cond, n_steps, method="midpoint"):
    """z_1 = z_pred + int_0^1 head dt."""
    z = z_pred
    ts = torch.linspace(0, 1, n_steps + 1, device=z_pred.device)
    for i in range(n_steps):
        t0 = ts[i].expand(z.shape[:2])
        dt = (ts[i + 1] - ts[i])
        if method == "euler":
            z = z + dt * head(z, t0, cond)
        else:  # midpoint
            k1 = head(z, t0, cond)
            zm = z + 0.5 * dt * k1
            tm = (ts[i] + 0.5 * dt).expand(z.shape[:2])
            z = z + dt * head(zm, tm, cond)
    return z


# ------------------------------------------------------------------ decode eval
def load_nova():
    from demo_nova3r import load_model as _load
    ck = str(REPO_ROOT / "nova3r_lib" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova, ncfg = _load(ck, DEV)
    nova.eval()
    for p in nova.parameters():
        p.requires_grad_(False)
    return nova, ncfg


def decode_cloud(nova, ncfg, tokens, pts_norm, num_q, seed):
    """Verbatim decode path from stitch_office4.decode()."""
    from nova3r.flow_matching.solver import ODESolver      # noqa: E402
    from nova3r.models.model_wrapper import BatchModelWrapper  # noqa: E402
    torch.manual_seed(seed)
    x_init = torch.rand(1, num_q, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04)
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x_init, method="midpoint", step_size=step,
                            return_intermediates=False,
                            images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": tokens.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (ta.query(b)[0].mean() + tb.query(a)[0].mean())


@torch.no_grad()
def decode_eval(head, nova, ncfg, bank, win_dirs, n_windows, n_steps, num_q, seed, tag):
    """Per window: does the flow move decode(z_pred) toward the oracle decode(z_tgt)?
    Reports Chamfer(flow, oracle) vs Chamfer(pred, oracle). Success = flow < pred."""
    idx = list(range(min(n_windows, len(win_dirs))))
    d_pred, d_flow = [], []
    for i in idx:
        pn = load_pts_norm(Path(bank["win_dirs"][i])).unsqueeze(0)
        z_pred = bank["z_pred"][i:i+1].to(DEV)
        z_tgt = bank["z_tgt"][i:i+1].to(DEV)
        cond = bank["cond"][i:i+1].to(DEV)
        z_flow = integrate(head, z_pred, cond, n_steps)
        c_pred = decode_cloud(nova, ncfg, z_pred, pn, num_q, seed)
        c_orac = decode_cloud(nova, ncfg, z_tgt, pn, num_q, seed)
        c_flow = decode_cloud(nova, ncfg, z_flow, pn, num_q, seed)
        d_pred.append(chamfer(c_pred, c_orac))
        d_flow.append(chamfer(c_flow, c_orac))
    d_pred, d_flow = np.array(d_pred), np.array(d_flow)
    win = int((d_flow < d_pred).sum())
    print(f"[decode-eval:{tag}] n={len(idx)}  "
          f"chamfer-to-oracle  pred={d_pred.mean():.4f}  flow={d_flow.mean():.4f}  "
          f"improved={win}/{len(idx)}  mean_delta={(d_pred-d_flow).mean()*100:+.2f}cm")
    return d_pred.mean(), d_flow.mean()


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--rooms", type=str, nargs="+", required=True)
    p.add_argument("--val-rooms", type=str, nargs="*", default=None)
    p.add_argument("--max-per-root", type=int, default=None)
    p.add_argument("--max-val-per-root", type=int, default=8)
    p.add_argument("--one-window-sanity", action="store_true")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-windows", type=int, default=6)
    p.add_argument("--num-queries", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-out", type=Path, default=None)
    p.add_argument("--no-decode-eval", action="store_true")
    return p.parse_args()


def resolve_windows(rooms, cap):
    win = []
    for root, ws in parse_root_configs(rooms):
        ws = ws if cap is None else ws[:cap]
        win.extend(ws)
    return win


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    adapter, acfg = load_frozen_adapter(args.base_ckpt)

    train_win = resolve_windows(args.rooms, args.max_per_root)
    if args.one_window_sanity:
        train_win = train_win[:1]
        print(f"[sanity] single window: {train_win[0]}")
    print(f"[data] {len(train_win)} train windows")
    train_bank = build_bank(adapter, train_win, "train")

    val_bank, val_win = None, None
    if args.val_rooms:
        val_win = resolve_windows(args.val_rooms, args.max_val_per_root)
        val_bank = build_bank(adapter, val_win, "val")

    head = TokenResidualFlow(token_dim=acfg["target_dim"], cond_dim=acfg["hidden_dim"],
                             hidden=args.hidden, depth=args.depth).to(DEV)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[head] TokenResidualFlow hidden={args.hidden} depth={args.depth}  params={n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    nova = ncfg = None
    if not args.no_decode_eval:
        nova, ncfg = load_nova()

    zP = train_bank["z_pred"].to(DEV)
    zT = train_bank["z_tgt"].to(DEV)
    cC = train_bank["cond"].to(DEV)
    N = zP.shape[0]
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    bs = min(args.batch_size, N)

    def run_eval(step):
        if args.no_decode_eval:
            return
        head.eval()
        decode_eval(head, nova, ncfg, train_bank, train_bank["win_dirs"],
                    args.eval_windows, args.flow_steps, args.num_queries, args.seed, f"train@{step}")
        if val_bank is not None:
            decode_eval(head, nova, ncfg, val_bank, val_bank["win_dirs"],
                        args.eval_windows, args.flow_steps, args.num_queries, args.seed, f"val@{step}")
        head.train()

    head.train()
    for step in range(1, args.steps + 1):
        sel = torch.randint(0, N, (bs,), device=DEV, generator=gen)
        loss = cfm_step(head, zP[sel], zT[sel], cC[sel], gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            print(f"[train] step {step}/{args.steps}  cfm_loss={loss.item():.5f}  lr={sched.get_last_lr()[0]:.2e}")
        if step % args.eval_every == 0:
            run_eval(step)

    run_eval(args.steps)
    if args.ckpt_out:
        args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head": head.state_dict(), "acfg": acfg,
                    "args": vars(args), "base_ckpt": str(args.base_ckpt)}, args.ckpt_out)
        print(f"[save] {args.ckpt_out}")


if __name__ == "__main__":
    main()
