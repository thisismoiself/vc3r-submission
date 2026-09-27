#!/usr/bin/env python3
"""
Interpolate consensus z_stars from training windows to fill the val window gap,
then decode and compare to the GT consensus in Rerun.

Interpolations tested:
  1. Linear:    equal-weight average of the two immediate neighbours
  2. Weighted:  inverse-temporal-distance weights across all training windows
  3. Quadratic: degree-2 polynomial through the 3 nearest training windows
  4. Cubic:     cubic spline through all training windows (scipy CubicSpline)

Usage:
  ! python scripts/interpolate_tokens.py --val-start 24 --n-left 3 --n-right 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.interpolate import CubicSpline
from numpy.polynomial import polynomial as P

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.inference import amp_dtype_mapping                   # noqa: E402

DATA_ROOT = REPO_ROOT / "scripts" / "data" / "windows"
N_FRAMES  = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-start",   type=int, default=24)
    p.add_argument("--n-left",      type=int, default=3)
    p.add_argument("--n-right",     type=int, default=3)
    p.add_argument("--num-queries", type=int, default=8192)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--rrd-out",     type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_window(start: int):
    win_dir   = DATA_ROOT / f"start_{start:04d}"
    meta      = torch.load(win_dir / "meta.pt",             map_location="cpu", weights_only=False)
    consensus = torch.load(win_dir / "z_star_consensus.pt", map_location="cpu", weights_only=True)
    pts_norm  = torch.load(win_dir / "pts_norm.pt",         map_location="cpu", weights_only=True)
    return meta, consensus, pts_norm


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
    args   = parse_args()
    device = torch.device(args.device)

    train_starts_left  = [args.val_start - (i + 1) * N_FRAMES
                          for i in range(args.n_left)][::-1]
    train_starts_right = [args.val_start + N_FRAMES + i * N_FRAMES
                          for i in range(args.n_right)]
    train_starts = train_starts_left + train_starts_right

    print(f"Val start: {args.val_start}  train: {train_starts}")

    # ── load all windows ───────────────────────────────────────────────────────
    train_z: dict[int, torch.Tensor] = {}
    for s in train_starts:
        meta, z, _ = load_window(s)
        train_z[s] = z[0]   # (768, 128)
        print(f"  loaded start={s:4d}  z_star l2={z.norm():.2f}")

    val_meta, val_z, val_pnorm = load_window(args.val_start)
    nf  = float(val_meta["norm_factor"])
    c2w = val_meta["poses_c2w"][0].numpy()

    # ── interpolation 1: immediate neighbours, equal weights ──────────────────
    left_neighbour  = train_starts_left[-1]   # closest left:  start=16
    right_neighbour = train_starts_right[0]   # closest right: start=32
    z_linear = 0.5 * train_z[left_neighbour] + 0.5 * train_z[right_neighbour]
    z_linear = z_linear.unsqueeze(0)   # (1, 768, 128)

    mse_linear = F.mse_loss(z_linear, val_z).item()
    print(f"\nLinear interp ({left_neighbour}+{right_neighbour})/2  "
          f"MSE vs GT: {mse_linear:.6f}")

    # ── interpolation 2: all windows, inverse temporal distance ───────────────
    dists   = {s: abs(s - args.val_start) for s in train_starts}
    weights = {s: 1.0 / d for s, d in dists.items()}
    total_w = sum(weights.values())
    z_weighted = sum(w / total_w * train_z[s]
                     for s, w in weights.items())
    z_weighted = z_weighted.unsqueeze(0)   # (1, 768, 128)

    mse_weighted = F.mse_loss(z_weighted, val_z).item()
    print(f"Weighted interp (1/dist across all {len(train_starts)} windows)  "
          f"MSE vs GT: {mse_weighted:.6f}")
    print(f"  weights: { {s: f'{w/total_w:.3f}' for s, w in weights.items()} }")

    # ── interpolation 3: quadratic through 3 nearest windows ──────────────────
    # Use closest left, closest right, and the next-closest on whichever side
    # has a second neighbour — here: 8, 16, 32, 40 → pick 16, 32, and (8 or 40)
    nearest3 = sorted(train_starts, key=lambda s: abs(s - args.val_start))[:3]
    nearest3.sort()
    x3  = np.array(nearest3, dtype=np.float64)
    y3  = np.stack([train_z[s].numpy().flatten() for s in nearest3])  # (3, D)
    # fit degree-2 polynomial per dimension (vectorised via lstsq)
    vander3 = np.vstack([x3**2, x3, np.ones(3)]).T   # (3, 3)
    coeffs3 = np.linalg.solve(vander3, y3)             # (3, D)
    val_x   = float(args.val_start)
    z_quad_np = coeffs3[0] * val_x**2 + coeffs3[1] * val_x + coeffs3[2]  # (D,)
    z_quad = torch.from_numpy(
        z_quad_np.reshape(1, 768, 128).astype(np.float32))

    mse_quad = F.mse_loss(z_quad, val_z).item()
    print(f"Quadratic interp (through starts {nearest3})  MSE vs GT: {mse_quad:.6f}")

    # ── interpolation 4: cubic spline through all training windows ────────────
    x_all = np.array(train_starts, dtype=np.float64)
    y_all = np.stack([train_z[s].numpy().flatten()
                      for s in train_starts])           # (N, D)
    cs    = CubicSpline(x_all, y_all)
    z_cubic_np = cs(val_x)                              # (D,)
    z_cubic = torch.from_numpy(
        z_cubic_np.reshape(1, 768, 128).astype(np.float32))

    mse_cubic = F.mse_loss(z_cubic, val_z).item()
    print(f"Cubic spline (all {len(train_starts)} windows)  MSE vs GT: {mse_cubic:.6f}")

    # ── load NOVA3R and decode all ─────────────────────────────────────────────
    print("\n[decode] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    print("[decode] GT consensus …")
    gt_norm       = decode_tokens(nova_model, nova_cfg, val_z,      val_pnorm,
                                  device, args.num_queries, seed=args.seed)
    print("[decode] Linear interpolation …")
    linear_norm   = decode_tokens(nova_model, nova_cfg, z_linear,   val_pnorm,
                                  device, args.num_queries, seed=args.seed)
    print("[decode] Weighted interpolation …")
    weighted_norm = decode_tokens(nova_model, nova_cfg, z_weighted, val_pnorm,
                                  device, args.num_queries, seed=args.seed)
    print("[decode] Quadratic interpolation …")
    quad_norm     = decode_tokens(nova_model, nova_cfg, z_quad,     val_pnorm,
                                  device, args.num_queries, seed=args.seed)
    print("[decode] Cubic spline …")
    cubic_norm    = decode_tokens(nova_model, nova_cfg, z_cubic,    val_pnorm,
                                  device, args.num_queries, seed=args.seed)

    gt_world       = norm_to_world(gt_norm,       nf, c2w)
    linear_world   = norm_to_world(linear_norm,   nf, c2w)
    weighted_world = norm_to_world(weighted_norm, nf, c2w)
    quad_world     = norm_to_world(quad_norm,     nf, c2w)
    cubic_world    = norm_to_world(cubic_norm,    nf, c2w)

    input_cam   = val_pnorm[0].numpy() / 3.0 * nf
    ones        = np.ones((len(input_cam), 1), dtype=np.float32)
    input_world = (c2w @ np.hstack([input_cam, ones]).T).T[:, :3]

    print(f"\n[centroid] GT:        {gt_world.mean(0).round(3)}")
    print(f"[centroid] Linear:    {linear_world.mean(0).round(3)}")
    print(f"[centroid] Weighted:  {weighted_world.mean(0).round(3)}")
    print(f"[centroid] Quadratic: {quad_world.mean(0).round(3)}")
    print(f"[centroid] Cubic:     {cubic_world.mean(0).round(3)}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    import rerun as rr

    if args.rrd_out is None:
        rrd_out = (REPO_ROOT / "outputs" / "consecutive_windows"
                   / f"interp_val{args.val_start}_L{args.n_left}R{args.n_right}.rrd")
    else:
        rrd_out = args.rrd_out
    rrd_out.parent.mkdir(parents=True, exist_ok=True)

    rr.init("interpolate_tokens", spawn=False)
    rr.save(str(rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    BLUE   = np.array([ 50, 150, 255], dtype=np.uint8)
    GREEN  = np.array([ 60, 220,  90], dtype=np.uint8)
    ORANGE = np.array([255, 140,  30], dtype=np.uint8)
    PINK   = np.array([220,  80, 200], dtype=np.uint8)
    RED    = np.array([220,  50,  50], dtype=np.uint8)
    GREY   = np.array([160, 160, 160], dtype=np.uint8)

    def sub(pts: np.ndarray, n: int = 100_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    rr.log("world/gt_consensus",
           rr.Points3D(sub(gt_world),
                       colors=np.tile(BLUE,   (min(len(gt_world),       100_000), 1)),
                       radii=0.012))
    rr.log("world/linear_interp",
           rr.Points3D(sub(linear_world),
                       colors=np.tile(GREEN,  (min(len(linear_world),   100_000), 1)),
                       radii=0.012))
    rr.log("world/weighted_interp",
           rr.Points3D(sub(weighted_world),
                       colors=np.tile(ORANGE, (min(len(weighted_world), 100_000), 1)),
                       radii=0.012))
    rr.log("world/quadratic_interp",
           rr.Points3D(sub(quad_world),
                       colors=np.tile(PINK,   (min(len(quad_world),     100_000), 1)),
                       radii=0.012))
    rr.log("world/cubic_spline",
           rr.Points3D(sub(cubic_world),
                       colors=np.tile(RED,    (min(len(cubic_world),    100_000), 1)),
                       radii=0.012))
    rr.log("world/input_pts",
           rr.Points3D(sub(input_world, n=30_000),
                       colors=np.tile(GREY,   (min(len(input_world),    30_000), 1)),
                       radii=0.008))

    rr.log("legend", rr.TextDocument(
        f"# Token interpolation — val start={args.val_start}\n\n"
        "- **Blue** `world/gt_consensus`: GT val consensus z_star decoded\n"
        f"- **Green** `world/linear_interp`: "
        f"(z_{left_neighbour} + z_{right_neighbour}) / 2  "
        f"→ MSE {mse_linear:.4f}\n"
        f"- **Orange** `world/weighted_interp`: "
        f"1/dist weighted across all {len(train_starts)} windows  "
        f"→ MSE {mse_weighted:.4f}\n"
        f"- **Pink** `world/quadratic_interp`: "
        f"degree-2 poly through {nearest3}  "
        f"→ MSE {mse_quad:.4f}\n"
        f"- **Red** `world/cubic_spline`: "
        f"cubic spline through all {len(train_starts)} windows  "
        f"→ MSE {mse_cubic:.4f}\n"
        "- **Grey** `world/input_pts`: val input pts (seed-0)\n\n"
        f"Train windows: {train_starts}\n"
        f"Val window: {args.val_start}–{args.val_start + N_FRAMES - 1}",
        media_type=rr.MediaType.MARKDOWN,
    ))

    print(f"\n[rerun] saved → {rrd_out}")
    print(f"  open with:  rerun {rrd_out}")


if __name__ == "__main__":
    main()
