#!/usr/bin/env python3
"""
Visualize one stride experiment.
Produces in viz/stride_<N>/:
  frames_grid.png        — 8 RGB frames
  pointcloud_frame_*.png — per-frame visible point cloud (colored by height)
  pointcloud_all.png     — all frames together, colored by frame index

Usage:
  python visualize_stride.py --stride 50
  python visualize_stride.py --stride 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DA3_SRC   = REPO_ROOT / "da3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from vc3r.replica import crop_visible_world_points  # noqa: E402


def render_pointcloud(pts: np.ndarray, color_vals: np.ndarray,
                      cmap: str, title: str, out_path: Path,
                      cam_pos: np.ndarray | None = None,
                      elev: float = 25, azim: float = -60,
                      vmin=None, vmax=None, cbar_label: str = ""):
    """Render a 3-D scatter with given color values; save to file."""
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    vmin = vmin if vmin is not None else np.percentile(color_vals, 2)
    vmax = vmax if vmax is not None else np.percentile(color_vals, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cols = matplotlib.colormaps[cmap](norm(color_vals))

    step = max(1, len(pts) // 20_000)   # cap at 20k points for speed
    sc = ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1],
                    c=cols[::step], s=0.5, linewidths=0, alpha=0.7)

    if cam_pos is not None:
        ax.scatter([cam_pos[0]], [cam_pos[2]], [cam_pos[1]],
                   color="red", s=80, zorder=10, marker="^", label="camera")
        ax.legend(fontsize=8)

    ax.set_xlabel("X (m)", fontsize=8); ax.set_ylabel("Z (m)", fontsize=8)
    ax.set_zlabel("Y (m)", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label(cbar_label, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    EXP_DIR  = REPO_ROOT / "experiments" / "overfit_8frames"
    data_dir = EXP_DIR / "data" / f"stride_{args.stride}"
    viz_dir  = EXP_DIR / "viz"  / f"stride_{args.stride}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    meta      = torch.load(data_dir / "meta.pt", weights_only=False)
    frame_ids = meta["frame_ids"]
    poses_c2w = meta["poses_c2w"]          # (T, 4, 4)
    K_native  = meta["K_native"].float()   # (3, 3)

    replica_root = Path(cfg.replica_root)
    room         = str(meta["room"])
    results_dir  = replica_root / room / "results"
    mesh_ply     = replica_root / f"{room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    depth_scale = float(cam["scale"])

    T      = len(frame_ids)
    colors = cm.tab10(np.linspace(0, 1, T))

    # ── mesh ─────────────────────────────────────────────────────────────────
    print(f"[viz stride={args.stride}] Loading mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    # ── 1. frames grid ───────────────────────────────────────────────────────
    print("[viz] Saving frames_grid.png …")
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    for ax, fid, col in zip(axes.flat, frame_ids, colors):
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        ax.imshow(img)
        ax.set_title(f"frame {fid:06d}", fontsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"8 sampled frames  (room0, stride={args.stride})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "frames_grid.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ── 2. per-frame point clouds ─────────────────────────────────────────────
    all_pts_world  = []
    all_frame_idx  = []

    for i, fid in enumerate(frame_ids):
        print(f"[viz] Frame {fid:06d} …", end=" ", flush=True)
        depth_img = Image.open(results_dir / f"depth{fid:06d}.png")
        depth = torch.from_numpy(
            np.asarray(depth_img, dtype=np.float32) / depth_scale
        )
        vis = crop_visible_world_points(
            points_world    = mesh_pts,
            camera_to_world = poses_c2w[i],
            intrinsics      = K_native,
            depth           = depth,
            depth_tolerance = args.depth_tolerance,
        )
        pts = vis["points_world"].numpy()   # (V, 3)  world coords XYZ
        print(f"{len(pts):,} pts")

        all_pts_world.append(pts)
        all_frame_idx.append(np.full(len(pts), i, dtype=np.int32))

        cam_pos = poses_c2w[i, :3, 3].numpy()

        # height coloring (Y axis)
        render_pointcloud(
            pts         = pts,
            color_vals  = pts[:, 1],
            cmap        = "plasma",
            title       = f"Frame {fid:06d} — visible pts colored by height (Y)",
            out_path    = viz_dir / f"pointcloud_frame{fid:06d}.png",
            cam_pos     = cam_pos,
            cbar_label  = "Y height (m)",
        )

    # ── 3. all frames together, colored by frame index ────────────────────────
    print("[viz] Saving pointcloud_all.png …")
    all_pts   = np.concatenate(all_pts_world, axis=0)
    all_fidx  = np.concatenate(all_frame_idx, axis=0)

    # Build discrete frame colors
    tab10 = matplotlib.colormaps["tab10"]
    point_colors = tab10(all_fidx / max(T - 1, 1))

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    step = max(1, len(all_pts) // 40_000)
    ax.scatter(all_pts[::step, 0], all_pts[::step, 2], all_pts[::step, 1],
               c=point_colors[::step], s=0.4, linewidths=0, alpha=0.6)

    # camera markers
    for i, fid in enumerate(frame_ids):
        cam = poses_c2w[i, :3, 3].numpy()
        ax.scatter(cam[0], cam[2], cam[1],
                   color=tab10(i / max(T - 1, 1)), s=120, marker="^",
                   zorder=10, edgecolors="k", linewidths=0.5)
        ax.text(cam[0], cam[2], cam[1] + 0.05, str(fid), fontsize=7,
                color=tab10(i / max(T - 1, 1)), ha="center")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y (m)")
    ax.view_init(elev=20, azim=-50)
    ax.set_title(f"All visible points — colored by frame  (stride={args.stride})",
                 fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    # legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=tab10(i / max(T - 1, 1)),
                               label=f"frame {fid:06d}")
               for i, fid in enumerate(frame_ids)]
    ax.legend(handles=patches, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), ncol=2)

    plt.tight_layout()
    plt.savefig(viz_dir / "pointcloud_all.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n[viz] Done — {viz_dir}/")
    for p in sorted(viz_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
