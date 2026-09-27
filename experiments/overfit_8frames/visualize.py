#!/usr/bin/env python3
"""
Visualize the 8 frames used in the overfit experiment.
Produces:
  viz/frames_grid.png         — 8 RGB frames in a 2×4 grid
  viz/pointclouds_grid.png    — per-frame visible points projected onto the frame image
  viz/pointcloud_topdown.png  — top-down view of all visible points (coloured by frame)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DA3_SRC   = REPO_ROOT / "da3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from vc3r.replica import crop_visible_world_points  # noqa: E402

# ── load meta ────────────────────────────────────────────────────────────────
meta       = torch.load(REPO_ROOT / "experiments/overfit_8frames/data/meta.pt",
                        weights_only=False)
frame_ids  = meta["frame_ids"]                     # list[int]
poses_c2w  = meta["poses_c2w"]                     # (T, 4, 4)
K_native   = meta["K_native"].float()              # (3, 3)
depth_tol  = float(meta.get("depth_tolerance", 0.05))

replica_root = Path("/storage/local/Replica")
room         = str(meta["room"])
room_dir     = replica_root / room
results_dir  = room_dir / "results"
mesh_ply     = replica_root / f"{room}_mesh.ply"

with (replica_root / "cam_params.json").open() as f:
    cam = json.load(f)["camera"]
depth_scale = float(cam["scale"])

viz_dir = REPO_ROOT / "experiments/overfit_8frames/viz"
viz_dir.mkdir(exist_ok=True)

T = len(frame_ids)
colors = cm.tab10(np.linspace(0, 1, T))

# ── sample mesh ───────────────────────────────────────────────────────────────
print("Loading mesh …")
mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))
print(f"  {mesh_pts.shape[0]:,} mesh points")

# ── per-frame data ────────────────────────────────────────────────────────────
rgb_frames   = []
vis_pts_cam  = []   # visible points in first-camera frame, per frame
vis_pts_world= []

first_pose_c2w = poses_c2w[0]
w2c_first      = torch.linalg.inv(first_pose_c2w)

def to_first_cam(pts_world):
    ones = torch.ones(pts_world.shape[0], 1)
    return (torch.cat([pts_world, ones], 1) @ w2c_first.T)[:, :3]

for i, fid in enumerate(frame_ids):
    print(f"Frame {fid:06d} …", end=" ")

    # RGB at native resolution for overlay, processed res for display
    img_native = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
    rgb_frames.append(img_native)

    # Depth at native resolution
    depth_img = Image.open(results_dir / f"depth{fid:06d}.png")
    depth = torch.from_numpy(np.asarray(depth_img, dtype=np.float32) / depth_scale)

    vis = crop_visible_world_points(
        points_world    = mesh_pts,
        camera_to_world = poses_c2w[i],
        intrinsics      = K_native,
        depth           = depth,
        depth_tolerance = depth_tol,
    )
    pts_w = vis["points_world"]   # (V, 3)
    uv    = vis["uv"]             # (V, 2)  pixel coords in native image

    vis_pts_world.append(pts_w)
    vis_pts_cam.append(to_first_cam(pts_w))
    print(f"{pts_w.shape[0]:,} visible pts")

# ── figure 1: RGB frames grid ─────────────────────────────────────────────────
print("Saving frames_grid.png …")
fig, axes = plt.subplots(2, 4, figsize=(20, 7))
for ax, img, fid, col in zip(axes.flat, rgb_frames, frame_ids, colors):
    ax.imshow(img)
    ax.set_title(f"frame {fid:06d}", color="black", fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(3)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("8 sampled frames  (room0, stride=20)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(viz_dir / "frames_grid.png", dpi=150, bbox_inches="tight")
plt.close()

# ── figure 2: per-frame point cloud overlay on image ─────────────────────────
print("Saving pointclouds_grid.png …")
fig, axes = plt.subplots(2, 4, figsize=(20, 7))

for ax, img, fid, col, i in zip(axes.flat, rgb_frames, frame_ids, colors, range(T)):
    # Reload depth for this frame
    depth_img = Image.open(results_dir / f"depth{fid:06d}.png")
    depth = torch.from_numpy(np.asarray(depth_img, dtype=np.float32) / depth_scale)

    vis = crop_visible_world_points(
        points_world    = mesh_pts,
        camera_to_world = poses_c2w[i],
        intrinsics      = K_native,
        depth           = depth,
        depth_tolerance = depth_tol,
    )
    uv = vis["uv"].numpy()    # (V, 2) — [u=col, v=row]

    ax.imshow(img)
    # Subsample for display speed
    step = max(1, len(uv) // 3000)
    ax.scatter(uv[::step, 0], uv[::step, 1],
               s=0.3, c=[col], alpha=0.6, linewidths=0)
    ax.set_title(f"frame {fid:06d}  ({len(uv):,} vis pts)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Visible GT mesh points overlaid on each frame", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(viz_dir / "pointclouds_grid.png", dpi=150, bbox_inches="tight")
plt.close()

# ── figure 3: top-down view of all visible points in world XZ plane ───────────
print("Saving pointcloud_topdown.png …")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i, (pts_w, col, fid) in enumerate(zip(vis_pts_world, colors, frame_ids)):
    pts_np = pts_w.numpy()
    step   = max(1, len(pts_np) // 4000)
    ax.scatter(pts_np[::step, 0], pts_np[::step, 2],
               s=0.5, color=col, alpha=0.5, label=f"frame {fid:06d}")
    # Camera position
    cam_pos = poses_c2w[i, :3, 3].numpy()
    ax.plot(cam_pos[0], cam_pos[2], "o", color=col, markersize=8, zorder=5)
    ax.annotate(f"{fid}", (cam_pos[0], cam_pos[2]),
                fontsize=7, ha="center", va="bottom", color=col)

ax.set_aspect("equal")
ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
ax.set_title("Top-down (XZ) view — visible points per frame", fontweight="bold")
ax.legend(loc="upper right", markerscale=8, fontsize=8)
plt.tight_layout()
plt.savefig(viz_dir / "pointcloud_topdown.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll saved to {viz_dir}/")
for p in sorted(viz_dir.iterdir()):
    print(f"  {p.name}")
