#!/usr/bin/env python3
"""
Train the DA3→NOVA3R adapter with a permutation-invariant set loss.

Uses Chamfer distance between the predicted token set and the GT consensus
token set. No pre-alignment of targets is needed — the loss is invariant to
any permutation of the 768 output tokens, so the adapter develops its own
internal slot semantics without being tied to a reference window.

Chamfer loss:
  For each predicted token, find its nearest GT token (p→t direction).
  For each GT token, find its nearest predicted token (t→p direction).
  Loss = mean of both directions.

This removes the reference-window dependency of the Hungarian pre-alignment
approach while keeping the same Q-Former architecture and training setup.

Usage:
  ! python scripts/train_setloss.py
  ! python scripts/train_setloss.py --val-start 56 --n-left 7 --n-right 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model                    # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper                  # noqa: E402
from nova3r.flow_matching.solver import ODESolver                          # noqa: E402
from nova3r.inference import amp_dtype_mapping                             # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402

DATA_ROOT = REPO_ROOT / "scripts" / "data" / "windows"
CFG_PATH  = OVERFIT_SRC / "config.yaml"
N_FRAMES  = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-start",   type=int,   default=56)
    p.add_argument("--n-left",      type=int,   default=7)
    p.add_argument("--n-right",     type=int,   default=7)
    p.add_argument("--steps",       type=int,   default=2000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--log-every",   type=int,   default=200)
    p.add_argument("--num-queries", type=int,   default=8192)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--rrd-out",     type=Path,  default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def window_starts(val_start: int, n_left: int, n_right: int):
    left  = [val_start - (i + 1) * N_FRAMES for i in range(n_left)][::-1]
    right = [val_start + N_FRAMES + i * N_FRAMES for i in range(n_right)]
    return left + right


def load_window(start: int):
    win_dir = DATA_ROOT / f"start_{start:04d}"
    if not win_dir.exists():
        raise FileNotFoundError(f"Window start={start} not cached.")
    meta      = torch.load(win_dir / "meta.pt",             map_location="cpu", weights_only=False)
    da3       = torch.load(win_dir / "da3_tokens.pt",       map_location="cpu", weights_only=True)
    consensus = torch.load(win_dir / "z_star_consensus.pt", map_location="cpu", weights_only=True)
    pts_norm  = torch.load(win_dir / "pts_norm.pt",         map_location="cpu", weights_only=True)
    return meta, da3, consensus, pts_norm


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (N, 768, 128)
    Symmetric Chamfer distance in token embedding space.
    Differentiable — gradients flow through the min operations.
    """
    dist  = torch.cdist(pred, target)                # (N, 768, 768)
    p2t   = dist.min(dim=2).values.mean()            # each pred → nearest GT
    t2p   = dist.min(dim=1).values.mean()            # each GT → nearest pred
    return (p2t + t2p) / 2


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor, pts_norm: torch.Tensor,
                  device: torch.device, num_queries: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, num_queries, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    use_amp = device.type != "cpu"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def norm_to_world(pts_norm_np: np.ndarray, norm_factor: float,
                  c2w: np.ndarray) -> np.ndarray:
    pts_cam = pts_norm_np / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(CFG_PATH)
    OmegaConf.set_struct(cfg, False)
    device = torch.device(args.device)

    train_starts = window_starts(args.val_start, args.n_left, args.n_right)

    print(f"[setup] val={args.val_start}  train={train_starts}  "
          f"(n_left={args.n_left}, n_right={args.n_right})")

    # ── load all windows (no pre-alignment) ───────────────────────────────────
    train_da3, train_z = [], []
    for s in train_starts:
        meta, da3, z, _ = load_window(s)
        train_da3.append(da3)
        train_z.append(z)
        print(f"  train start={s:4d}  frames {meta['frame_ids'][0]}–{meta['frame_ids'][-1]}  "
              f"norm_factor={float(meta['norm_factor']):.3f}m")

    val_meta, val_da3, val_z, val_pnorm = load_window(args.val_start)
    print(f"  val   start={args.val_start:4d}  "
          f"frames {val_meta['frame_ids'][0]}–{val_meta['frame_ids'][-1]}  "
          f"norm_factor={float(val_meta['norm_factor']):.3f}m")

    da3_train   = torch.cat(train_da3, dim=0).to(device)   # (N, T, D)
    zstar_train = torch.cat(train_z,   dim=0).to(device)   # (N, 768, 128)
    val_da3_dev = val_da3.to(device)
    val_z_dev   = val_z.to(device)
    N = da3_train.shape[0]

    # ── adapter ───────────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
        drop          = 0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n[train] {N} windows  {args.steps} steps  lr={args.lr}  "
          f"(Chamfer set loss, no pre-alignment)")
    adapter.train()
    best_val_loss  = float("inf")
    best_step      = 0
    best_state     = None

    for step in range(1, args.steps + 1):
        idx  = torch.randperm(N, generator=rng)
        pred = adapter(da3_train[idx])
        loss = chamfer_loss(pred, zstar_train[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                val_loss = chamfer_loss(adapter(val_da3_dev), val_z_dev).item()
            marker = " ← best" if val_loss < best_val_loss else ""
            print(f"  step {step:5d}/{args.steps}  "
                  f"train_loss={loss.item():.6f}  val_loss={val_loss:.6f}{marker}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step     = step
                best_state    = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}

    print(f"\n[best]  step {best_step}  val Chamfer {best_val_loss:.6f}")
    adapter.load_state_dict(best_state)
    adapter.eval()

    ckpt_path = (args.rrd_out.parent if args.rrd_out else
                 REPO_ROOT / "outputs" / "consecutive_windows") / \
                f"setloss_val{args.val_start}_L{args.n_left}R{args.n_right}_best.pt"
    torch.save({"state_dict": best_state, "step": best_step, "val_loss": best_val_loss},
               ckpt_path)
    print(f"[best]  saved → {ckpt_path}")

    with torch.no_grad():
        pred_tokens = adapter(val_da3_dev).cpu()

    final_train = chamfer_loss(adapter(da3_train).cpu(), zstar_train.cpu()).item()
    final_val   = chamfer_loss(pred_tokens, val_z.cpu()).item()
    print(f"[result] train Chamfer: {final_train:.6f}  val Chamfer: {final_val:.6f}  "
          f"(best model from step {best_step})")

    # ── NOVA3R decode ──────────────────────────────────────────────────────────
    print("[decode] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    nf  = float(val_meta["norm_factor"])
    c2w = val_meta["poses_c2w"][0].numpy()

    print("[decode] GT consensus z_star …")
    gt_norm   = decode_tokens(nova_model, nova_cfg, val_z.cpu(), val_pnorm,
                              device, args.num_queries, seed=args.seed)
    print("[decode] Adapter prediction …")
    pred_norm = decode_tokens(nova_model, nova_cfg, pred_tokens, val_pnorm,
                              device, args.num_queries, seed=args.seed)

    gt_world   = norm_to_world(gt_norm,   nf, c2w)
    pred_world = norm_to_world(pred_norm, nf, c2w)
    input_cam  = val_pnorm[0].numpy() / 3.0 * nf
    ones       = np.ones((len(input_cam), 1), dtype=np.float32)
    input_world = (c2w @ np.hstack([input_cam, ones]).T).T[:, :3]

    print(f"[decode] GT centroid:   {gt_world.mean(0).round(3)}")
    print(f"[decode] Pred centroid: {pred_world.mean(0).round(3)}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    import rerun as rr

    if args.rrd_out is None:
        tag     = f"setloss_val{args.val_start}_L{args.n_left}R{args.n_right}"
        rrd_out = REPO_ROOT / "outputs" / "consecutive_windows" / f"{tag}.rrd"
    else:
        rrd_out = args.rrd_out
    rrd_out.parent.mkdir(parents=True, exist_ok=True)

    rr.init("setloss_windows", spawn=False)
    rr.save(str(rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    BLUE   = np.array([ 30, 120, 255], dtype=np.uint8)
    GREEN  = np.array([ 50, 210,  80], dtype=np.uint8)
    GREY   = np.array([160, 160, 160], dtype=np.uint8)
    ORANGE = np.array([255, 160,   0], dtype=np.uint8)

    def sub(pts: np.ndarray, n: int = 100_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    rr.log("world/gt_consensus",
           rr.Points3D(sub(gt_world),
                       colors=np.tile(BLUE,  (min(len(gt_world),   100_000), 1)),
                       radii=0.012))
    rr.log("world/adapter_pred",
           rr.Points3D(sub(pred_world),
                       colors=np.tile(GREEN, (min(len(pred_world), 100_000), 1)),
                       radii=0.012))
    rr.log("world/input_pts",
           rr.Points3D(sub(input_world, n=30_000),
                       colors=np.tile(GREY,  (min(len(input_world), 30_000), 1)),
                       radii=0.008))
    rr.log("world/cameras",
           rr.Points3D(val_meta["poses_c2w"][:, :3, 3].numpy(),
                       colors=np.tile(ORANGE, (len(val_meta["poses_c2w"]), 1)),
                       radii=0.04))

    train_frame_ranges = " | ".join(f"{s}–{s+N_FRAMES-1}" for s in train_starts)
    rr.log("legend", rr.TextDocument(
        f"# Chamfer set loss — val start={args.val_start}\n\n"
        "- **Blue** `world/gt_consensus`: GT val consensus z_star\n"
        "- **Green** `world/adapter_pred`: adapter prediction\n"
        "- **Grey** `world/input_pts`: val input pts (seed-0)\n"
        "- **Orange** `world/cameras`: val window cameras\n\n"
        f"**Train frames:** {train_frame_ranges}\n\n"
        f"**Val frames:** {args.val_start}–{args.val_start+N_FRAMES-1}\n\n"
        f"N={N} windows  |  Steps={args.steps}  |  LR={args.lr}\n\n"
        f"Train Chamfer: {final_train:.4f}  |  Val Chamfer: {final_val:.4f}",
        media_type=rr.MediaType.MARKDOWN,
    ))

    print(f"\n[rerun] saved → {rrd_out}")
    print(f"  open with:  rerun {rrd_out}")


if __name__ == "__main__":
    main()
