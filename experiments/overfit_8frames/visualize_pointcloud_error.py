#!/usr/bin/env python3
"""Visualize adapter error in decoded point-cloud space."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "overfit_8frames"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P = NOVA3R_ROOT / "third_party"
DA3_SRC = REPO_ROOT / "da3" / "src"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(EXP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from multi_scene_train import decode_tokens, chamfer  # noqa: E402


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for p, c in zip(xyz.astype(np.float32), rgb8):
            f.write(struct.pack("<fffBBB", p[0], p[1], p[2], c[0], c[1], c[2]))


def error_colours(values: np.ndarray, vmax: float) -> np.ndarray:
    cmap = matplotlib.colormaps["magma"]
    rgba = cmap(Normalize(vmin=0.0, vmax=vmax, clip=True)(values))
    return (rgba[:, :3] * 255).astype(np.uint8)


def axis_equal(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=25, azim=-60)


def scatter_cloud(ax, pts: np.ndarray, colors, title: str, size: float = 3.0):
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=colors, s=size, linewidths=0, alpha=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    axis_equal(ax, pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--num-queries", type=int, default=768)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)

    da3_tokens = torch.load(args.sample_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    z_star = torch.load(args.sample_dir / "z_star.pt", map_location="cpu", weights_only=True)
    pts_norm = torch.load(args.sample_dir / "pts_norm.pt", map_location="cpu", weights_only=True)
    meta = torch.load(args.sample_dir / "meta.pt", map_location="cpu", weights_only=False)

    print("[viz_error] Loading adapter ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim),
        hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens),
        target_dim=int(cfg.target_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()

    print("[viz_error] Loading NOVA3R ...")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    print("[viz_error] Decoding target and prediction ...")
    with torch.no_grad():
        pred_tokens = adapter(da3_tokens.to(device)).cpu()
    gt_dec = decode_tokens(nova_model, nova_cfg, z_star, pts_norm, device, args.num_queries, seed=42)[0]
    pred_dec = decode_tokens(nova_model, nova_cfg, pred_tokens, pts_norm, device, args.num_queries, seed=42)[0]

    # Predicted-point error: distance from each predicted point to nearest target decoded point.
    d_pred_to_gt = torch.cdist(pred_dec.float().unsqueeze(0), gt_dec.float().unsqueeze(0))[0].min(dim=1).values
    d_gt_to_pred = torch.cdist(gt_dec.float().unsqueeze(0), pred_dec.float().unsqueeze(0))[0].min(dim=1).values

    pred_np = pred_dec.numpy()
    gt_np = gt_dec.numpy()
    input_np = pts_norm[0].numpy()
    err = d_pred_to_gt.numpy()
    err_gt = d_gt_to_pred.numpy()
    cd_pg = chamfer(pred_dec, gt_dec)
    cd_ai = chamfer(gt_dec, pts_norm[0])
    mse = float(torch.nn.functional.mse_loss(pred_tokens, z_star))

    p95 = float(np.percentile(err, 95))
    p50 = float(np.percentile(err, 50))
    p90 = float(np.percentile(err, 90))
    vmax = max(p95, 1e-6)
    pred_rgb = error_colours(err, vmax)
    gt_rgb = np.full((len(gt_np), 3), 180, dtype=np.uint8)
    input_rgb = np.full((len(input_np), 3), 150, dtype=np.uint8)

    write_ply(args.out_dir / "input_cloud.ply", input_np, input_rgb)
    write_ply(args.out_dir / "gt_decoded.ply", gt_np, gt_rgb)
    write_ply(args.out_dir / "pred_decoded_error_colored.ply", pred_np, pred_rgb)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        "Decoded Point-Cloud Error: Predicted NOVA3R Tokens vs Target z_star Decode\n"
        f"frames={meta['frame_ids']}  MSE={mse:.4f}  CD(pred,gt)={cd_pg:.4f}  "
        f"CD(AE,input)={cd_ai:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.78],
        height_ratios=[1.0, 1.0],
    )

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    scatter_cloud(ax1, input_np, input_rgb / 255.0, "Input normalized point cloud", size=1.2)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    scatter_cloud(ax2, gt_np, "#4c78a8", "Target: NOVA3R z_star decode", size=4.0)

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    scatter_cloud(
        ax3,
        pred_np,
        pred_rgb / 255.0,
        f"Prediction colored by nearest-target error\nmedian={p50:.4f}, p90={p90:.4f}, p95={p95:.4f}",
        size=4.0,
    )

    ax4 = fig.add_subplot(gs[1, 0], projection="3d")
    ax4.scatter(gt_np[:, 0], gt_np[:, 2], gt_np[:, 1], c="#b8b8b8", s=5, linewidths=0, alpha=0.35)
    ax4.scatter(pred_np[:, 0], pred_np[:, 2], pred_np[:, 1], c=pred_rgb / 255.0, s=5, linewidths=0, alpha=0.85)
    ax4.set_title("Overlay: target grey, prediction error-colored", fontsize=10, fontweight="bold")
    axis_equal(ax4, np.concatenate([gt_np, pred_np], axis=0))

    ax5 = fig.add_subplot(gs[1, 1:3])
    ax5.hist(err, bins=40, color="#c43c39", alpha=0.85)
    ax5.axvline(p50, color="black", lw=1.5, label=f"median {p50:.4f}")
    ax5.axvline(p90, color="#333333", lw=1.2, ls="--", label=f"p90 {p90:.4f}")
    ax5.axvline(p95, color="#777777", lw=1.2, ls=":", label=f"p95 {p95:.4f}")
    ax5.set_title("Predicted point nearest-target distances", fontsize=10, fontweight="bold")
    ax5.set_xlabel("distance")
    ax5.set_ylabel("count")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[:, 3])
    stats_text = (
        f"Token MSE: {mse:.6f}\n"
        f"Chamfer pred <-> gt: {cd_pg:.6f}\n"
        f"Chamfer AE <-> input: {cd_ai:.6f}\n\n"
        f"Pred -> target NN distances:\n"
        f"  mean: {err.mean():.6f}\n"
        f"  median: {p50:.6f}\n"
        f"  p90: {p90:.6f}\n"
        f"  p95: {p95:.6f}\n"
        f"  max: {err.max():.6f}\n\n"
        f"Target -> pred NN distances:\n"
        f"  mean: {err_gt.mean():.6f}\n"
        f"  median: {np.median(err_gt):.6f}\n"
        f"  p90: {np.percentile(err_gt, 90):.6f}\n"
        f"  p95: {np.percentile(err_gt, 95):.6f}"
    )
    ax6.axis("off")
    ax6.text(0.02, 0.98, stats_text, va="top", ha="left", family="monospace", fontsize=10)

    mappable = matplotlib.cm.ScalarMappable(norm=Normalize(0.0, vmax), cmap="magma")
    fig.colorbar(mappable, ax=[ax3, ax4], shrink=0.8, label="Predicted point distance to nearest target")
    out_png = args.out_dir / "pointcloud_error_summary.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics_path = args.out_dir / "metrics.txt"
    metrics_path.write_text(stats_text + "\n", encoding="utf-8")
    print(f"[viz_error] Wrote {out_png}")
    print(f"[viz_error] Wrote {metrics_path}")
    print(f"[viz_error] Wrote PLY files in {args.out_dir}")


if __name__ == "__main__":
    main()
