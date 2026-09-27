#!/usr/bin/env python3
"""Point-space flow-matching corrector on top of a FROZEN MSE adapter + decoder.

Motivation: the token-space residual flow (train_residual_flow.py) FAILED at scale --
lowering token loss 12x made the decode WORSE, because token error is decode-amplified
(~6x geometric damage, see token-noise sensitivity). This corrector instead operates
DIRECTLY in 3D point space, downstream of the decoder, so there is no decoder in between
to amplify error: moving a point 1cm toward the surface makes the cloud 1cm better.

Pipeline:
  DA3 tokens --[frozen adapter]--> z_pred --[frozen decoder]--> P_pred (predicted cloud)
  then a conditional flow-matching CORRECTOR transports P_pred -> the GT surface,
  conditioned ONLY on z_pred tokens (available at deploy; no GT input).

  Target = GT visible cloud (pts_norm). This can EXCEED the oracle ceiling (FURN F@2 0.551)
  because the corrector is not a decoder -- it is not bound by what the frozen decoder can
  express from tokens; it learns "given the scene tokens + a roughly-placed point, where is
  the real surface". Validation is office4 (held out): GT is used ONLY to score, never as
  input, so a val improvement is a genuine GT-free generalization result.

Flow: rectified/straight CFM between the two point distributions with minibatch-OT coupling
(lap.lapjv). x_t=(1-t)x0+t x1, v*=x1-x0; x0~P_pred, x1~GT, coupled by OT. Inference
integrates the actual P_pred points forward (midpoint, few steps).

Metric: per-window Chamfer / F@2cm / F@5cm to GT in METERS (scale = norm_factor/3), for the
corrected cloud vs the P_pred baseline, on train AND val windows.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(REPO_ROOT / "scripts"), str(REPO_ROOT / "da3" / "src"),
           str(REPO_ROOT / "nova3r_lib" / "third_party"), str(REPO_ROOT / "nova3r_lib")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_online_var_hungarian import (          # noqa: E402
    parse_root_configs, load_window, load_pts_norm,
)
# Re-assert nova3r_lib ahead of the bare nova3r/ copy the trainer import prepends.
for _p in [str(REPO_ROOT / "nova3r_lib" / "third_party"), str(REPO_ROOT / "nova3r_lib")]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from train_residual_flow import (                 # noqa: E402
    load_frozen_adapter, adapter_pred_and_feats, load_nova, decode_cloud,
)
from vc3r.alignment import CrossAttentionBlock  # noqa: E402

try:
    import lap
    def solve_ot(cost):                      # x[i] = column (GT idx) assigned to source i
        return lap.lapjv(cost, extend_cost=True)[1]
except ImportError:
    from scipy.optimize import linear_sum_assignment
    def solve_ot(cost):
        return linear_sum_assignment(cost)[1]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ bank
def build_point_bank(adapter, nova, ncfg, win_dirs, cache_dir: Path, num_q, seed, tag):
    """Per window: P_pred = decode(adapter(da3)), z_pred tokens, gt=pts_norm, scale=nf/3.
    Decoding is the expensive step, so the bank is cached to disk and reused."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i, wd in enumerate(win_dirs):
        key = f"{wd.parent.name}__{wd.name}__q{num_q}.pt"
        cp = cache_dir / key
        if cp.exists():
            items.append(cp)
            continue
        meta, da3, mean, _ = load_window(wd)
        z_pred, _ = adapter_pred_and_feats(adapter, da3.unsqueeze(0) if da3.ndim == 2 else da3)
        p_pred = decode_cloud(nova, ncfg, z_pred, load_pts_norm(wd).unsqueeze(0), num_q, seed)
        scale = float(meta["norm_factor"]) / 3.0
        torch.save({"p_pred": torch.from_numpy(p_pred).float(),
                    "z_pred": z_pred[0].cpu(),
                    "gt": load_pts_norm(wd),
                    "scale": scale, "win": str(wd)}, cp)
        items.append(cp)
        if (i + 1) % 25 == 0 or i + 1 == len(win_dirs):
            print(f"[bank:{tag}] {i+1}/{len(win_dirs)} decoded", flush=True)
    return items


def load_bank_item(cp):
    d = torch.load(cp, map_location="cpu", weights_only=False)
    return d["p_pred"], d["z_pred"], d["gt"], d["scale"]


# ------------------------------------------------------------------ corrector
class PointFlowCorrector(torch.nn.Module):
    """Conditional velocity field v(x_t, t | z_pred tokens). Points are queries that
    self-attend and cross-attend to the projected token memory (reusing the adapter's
    CrossAttentionBlock). Time is injected into the point embedding."""

    def __init__(self, token_dim=128, hidden=256, depth=4, num_heads=8, fourier_bands=8,
                 time_bands=64, local_knn=0):
        super().__init__()
        self.fb = fourier_bands
        self.tb = time_bands
        self.local_knn = local_knn
        self.point_in = torch.nn.Linear(3 * 2 * fourier_bands, hidden)
        self.time_in = torch.nn.Linear(2 * time_bands, hidden)
        self.token_proj = torch.nn.Linear(token_dim, hidden)
        if local_knn > 0:
            # PointNet over each query point's k-NN offsets in P_pred: gives the field the
            # LOCAL surface structure (orientation/density) so it can snap points ONTO the
            # surface instead of inferring it from the global token memory alone.
            self.local_encoder = torch.nn.Sequential(
                torch.nn.Linear(4, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden))
        self.blocks = torch.nn.ModuleList(
            [CrossAttentionBlock(dim=hidden, num_heads=num_heads) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(hidden)
        self.out = torch.nn.Linear(hidden, 3)
        self.out.weight.data.mul_(0.1); self.out.bias.data.zero_()   # near-zero velocity at init

    def _fourier(self, x, bands):
        freqs = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * math.pi
        a = x[..., None] * freqs
        return torch.cat([a.sin(), a.cos()], dim=-1).reshape(*x.shape[:-1], -1)

    def forward(self, x_t, t, tokens, nbr=None):
        # x_t [B,N,3], t [B,N], tokens [B,768,token_dim], nbr [B,N,k,3] local offsets (or None)
        h = self.point_in(self._fourier(x_t, self.fb)) + self.time_in(self._fourier(t[..., None], self.tb))
        if self.local_knn > 0 and nbr is not None:
            # per-neighbour features [offset_xyz, distance] -> shared MLP -> max-pool over k
            d = nbr.norm(dim=-1, keepdim=True)
            lf = self.local_encoder(torch.cat([nbr, d], dim=-1)).amax(dim=2)   # [B,N,hidden]
            h = h + lf
        mem = self.token_proj(tokens)
        for blk in self.blocks:
            h = blk(h, mem)
        return self.out(self.norm(h))


# ------------------------------------------------------------------ local k-NN geometry
def knn_offsets_gpu(query, ref, k, drop_self):
    """Per query point, offsets to its k nearest neighbours in `ref`. GPU cdist (small clouds).
    query [B,n,3], ref [B,M,3] -> [B,n,k,3]. drop_self: skip the 0-distance self match."""
    d = torch.cdist(query, ref)                                  # [B,n,M]
    idx = d.topk(k + (1 if drop_self else 0), dim=-1, largest=False).indices
    if drop_self:
        idx = idx[..., 1:]                                       # drop nearest (self)
    nbr = torch.gather(ref.unsqueeze(1).expand(-1, query.shape[1], -1, -1), 2,
                       idx.unsqueeze(-1).expand(-1, -1, -1, 3))   # [B,n,k,3]
    return nbr - query.unsqueeze(2)


def knn_offsets_cpu(cloud, k):
    """k-NN offsets of every point in `cloud` to its neighbours within the SAME cloud, via
    cKDTree (memory-safe for large inference clouds e.g. 50k). cloud [M,3] -> [M,k,3]."""
    c = cloud.detach().cpu().numpy()
    _, idx = cKDTree(c).query(c, k=k + 1)                        # includes self at col 0
    nbr = c[idx[:, 1:]] - c[:, None, :]                          # [M,k,3]
    return torch.from_numpy(nbr).float()


# ------------------------------------------------------------------ training
def nn_flow_loss(model, p_pred, gt, tokens, n_couple, gen):
    """Nearest-neighbour-coupled straight flow. Each sampled prediction point is coupled to
    its NEAREST GT surface point -> a DETERMINISTIC, low-variance target ("snap the point to
    the nearest surface"). This is the fix for random minibatch-OT, whose fresh-per-step
    coupling made v* high-variance so the L2-optimum collapsed to ~zero velocity (identity).
    NN coupling gives a well-defined field that improves accuracy without collapsing."""
    B = p_pred.shape[0]
    si = torch.randint(p_pred.shape[1], (B, n_couple), device=DEV, generator=gen)
    x0 = torch.gather(p_pred, 1, si[..., None].expand(-1, -1, 3))     # [B, n, 3]
    nn = torch.cdist(x0, gt).argmin(dim=-1)                          # nearest GT idx, [B, n]
    x1 = torch.gather(gt, 1, nn[..., None].expand(-1, -1, 3))
    t = torch.rand(B, n_couple, device=DEV, generator=gen)
    x_ti = (1 - t)[..., None] * x0 + t[..., None] * x1
    v_star = x1 - x0
    nbr = knn_offsets_gpu(x0, p_pred, model.local_knn, drop_self=True) if model.local_knn > 0 else None
    return (model(x_ti, t, tokens, nbr) - v_star).square().mean()


@torch.no_grad()
def integrate(model, x0, tokens, n_steps):
    x = x0
    # local geometry features are a FIXED per-point descriptor of the initial P_pred cloud
    # (its local surface structure), computed once via cKDTree so 50k-point clouds stay memory-safe.
    nbr = None
    if getattr(model, "local_knn", 0) > 0:
        nbr = torch.stack([knn_offsets_cpu(x0[b], model.local_knn) for b in range(x0.shape[0])]).to(DEV)
    ts = torch.linspace(0, 1, n_steps + 1, device=DEV)
    for i in range(n_steps):
        t0 = ts[i].expand(x.shape[:2]); dt = ts[i + 1] - ts[i]
        k1 = model(x, t0, tokens, nbr)
        xm = x + 0.5 * dt * k1
        tm = (ts[i] + 0.5 * dt).expand(x.shape[:2])
        x = x + dt * model(xm, tm, tokens, nbr)
    return x


def eval_cloud_m(pred, gt, scale, thresholds=(0.02, 0.05)):
    """Chamfer + F@thresh in METERS (clouds scaled by nf/3)."""
    p, g = pred * scale, gt * scale
    tp, tg = cKDTree(p), cKDTree(g)
    d_pg = tp.query(g)[0]      # completeness: gt -> pred
    d_gp = tg.query(p)[0]      # accuracy:     pred -> gt   (query gt-tree with pred)
    acc, comp = d_gp.mean(), d_pg.mean()
    out = {"chamfer": 0.5 * (acc + comp), "acc": acc, "comp": comp}
    for th in thresholds:
        prec = (d_gp < th).mean(); rec = (d_pg < th).mean()
        out[f"F@{int(th*100)}"] = 2 * prec * rec / (prec + rec + 1e-9)
    return out


@torch.no_grad()
def decode_eval(model, bank_items, n_steps, n_windows, tag):
    idx = list(range(min(n_windows, len(bank_items))))
    agg = {"pred": [], "flow": []}
    for i in idx:
        p_pred, z_pred, gt, scale = load_bank_item(bank_items[i])
        x0 = p_pred.unsqueeze(0).to(DEV); tok = z_pred.unsqueeze(0).to(DEV)
        corr = integrate(model, x0, tok, n_steps)[0].cpu().numpy()
        agg["pred"].append(eval_cloud_m(p_pred.numpy(), gt.numpy(), scale))
        agg["flow"].append(eval_cloud_m(corr, gt.numpy(), scale))
    def mean(k, m): return float(np.mean([a[m] for a in agg[k]]))
    imp = sum(f["chamfer"] < p["chamfer"] for p, f in zip(agg["pred"], agg["flow"]))
    print(f"[eval:{tag}] n={len(idx)}  "
          f"chamfer pred={mean('pred','chamfer')*100:.2f}cm flow={mean('flow','chamfer')*100:.2f}cm  "
          f"F@2 pred={mean('pred','F@2'):.3f} flow={mean('flow','F@2'):.3f}  "
          f"F@5 pred={mean('pred','F@5'):.3f} flow={mean('flow','F@5'):.3f}  "
          f"improved={imp}/{len(idx)}", flush=True)
    return mean('flow', 'F@2')


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--rooms", type=str, nargs="+", required=True)
    p.add_argument("--val-rooms", type=str, nargs="*", default=None)
    p.add_argument("--max-per-root", type=int, default=None)
    p.add_argument("--max-val-per-root", type=int, default=8)
    p.add_argument("--bank-queries", type=int, default=8192)
    p.add_argument("--bank-dir", type=Path, default=REPO_ROOT / "outputs" / "point_flow_bank")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-couple", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--local-knn", type=int, default=0,
                   help="k for local geometry conditioning (0=off). Each query point sees its k "
                        "nearest P_pred neighbours (offsets+distance) via a PointNet, giving the "
                        "corrector local surface structure. Only changes the corrector, not the decoder.")
    p.add_argument("--flow-steps", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-out", type=Path, default=None)
    return p.parse_args()


def resolve_windows(rooms, cap):
    win = []
    for root, ws in parse_root_configs(rooms):
        win.extend(ws if cap is None else ws[:cap])
    return win


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    adapter, acfg = load_frozen_adapter(args.base_ckpt)
    nova, ncfg = load_nova()

    train_win = resolve_windows(args.rooms, args.max_per_root)
    print(f"[data] {len(train_win)} train windows; building/loading P_pred bank...", flush=True)
    train_bank = build_point_bank(adapter, nova, ncfg, train_win,
                                  args.bank_dir, args.bank_queries, args.seed, "train")
    val_bank = None
    if args.val_rooms:
        val_win = resolve_windows(args.val_rooms, args.max_val_per_root)
        val_bank = build_point_bank(adapter, nova, ncfg, val_win,
                                    args.bank_dir, args.bank_queries, args.seed, "val")

    model = PointFlowCorrector(token_dim=acfg["target_dim"], hidden=args.hidden, depth=args.depth,
                               local_knn=args.local_knn).to(DEV)
    print(f"[model] PointFlowCorrector hidden={args.hidden} depth={args.depth} "
          f"local_knn={args.local_knn}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # preload banks to RAM (P_pred + z_pred are small)
    T = [load_bank_item(cp) for cp in train_bank]
    P = torch.stack([t[0] for t in T]).to(DEV)      # [N, Q, 3]
    Z = torch.stack([t[1] for t in T]).to(DEV)      # [N, 768, 128]
    G = torch.stack([t[2] for t in T]).to(DEV)      # [N, K, 3]
    N = P.shape[0]
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    bs = min(args.batch_size, N)

    def run_eval(step):
        model.eval()
        decode_eval(model, train_bank, args.flow_steps, args.eval_windows, f"train@{step}")
        if val_bank is not None:
            decode_eval(model, val_bank, args.flow_steps, args.eval_windows, f"val@{step}")
        model.train()

    model.train()
    for step in range(1, args.steps + 1):
        sel = torch.randint(0, N, (bs,), device=DEV, generator=gen)
        loss = nn_flow_loss(model, P[sel], G[sel], Z[sel], args.n_couple, gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            print(f"[train] step {step}/{args.steps}  ot_flow_loss={loss.item():.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        if step % args.eval_every == 0:
            run_eval(step)
    run_eval(args.steps)
    if args.ckpt_out:
        args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "acfg": acfg, "args": vars(args)}, args.ckpt_out)
        print(f"[save] {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
