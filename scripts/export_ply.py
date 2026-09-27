#!/usr/bin/env python3
"""Export the progress-strip frames as real coloured .ply point clouds for interactive inspection.
Reads base.npz (observed + GT occluded surfaces) and a frame npz (model-filled surface), converts
voxel indices -> metric xyz, and writes ASCII PLY with per-point colour.

  blue   = observed input (partial scan)
  orange = model-filled occluded voxels (the completion)
  green  = ground-truth occluded surface
"""
import argparse, glob
from pathlib import Path
import numpy as np


def pts(grid, voxel):
    return np.argwhere(grid).astype(np.float32) * voxel


def write_ply(path, clouds):
    """clouds: list of (Nx3 xyz, (r,g,b))."""
    xyz = np.concatenate([c for c, _ in clouds], 0) if clouds else np.zeros((0, 3))
    rgb = np.concatenate([np.tile(np.array(col, np.uint8), (len(c), 1)) for c, col in clouds], 0) \
        if clouds else np.zeros((0, 3), np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"-> {path}  ({len(xyz)} pts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip-dir", default="/usr/prakt/s0016/vc3r/experiments/overfit_8frames/progress_strip")
    ap.add_argument("--step", type=int, default=-1, help="which milestone frame; -1 = latest")
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/experiments/overfit_8frames/progress_strip/ply")
    args = ap.parse_args()

    d = Path(args.strip_dir)
    base = np.load(d / "base.npz")
    voxel = float(base["voxel"])
    obs = pts(base["obs"], voxel)           # blue
    gt = pts(base["gt_new"], voxel)          # green

    frames = sorted(glob.glob(str(d / "frames" / "step_*.npz")), key=lambda p: int(Path(p).stem.split("_")[1]))
    fp = frames[-1] if args.step < 0 else str(d / "frames" / f"step_{args.step:06d}.npz")
    fr = np.load(fp); step = int(fr["step"]); model = pts(fr["pred_new"], voxel)  # orange
    print(f"[frame] step {step}: obs={len(obs)} model={len(model)} gt={len(gt)} "
          f"recall={float(fr['recall']):.1f}% MAE={float(fr['mae']):.2f}cm  voxel={voxel*100:.1f}cm")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    BLUE, ORANGE, GREEN = (60, 130, 210), (255, 130, 30), (30, 200, 90)
    # 1) model completion: observed + model-filled
    write_ply(out / f"completion_step{step//1000}k.ply", [(obs, BLUE), (model, ORANGE)])
    # 2) ground truth: observed + GT occluded
    write_ply(out / "ground_truth.ply", [(obs, BLUE), (gt, GREEN)])
    # 3) overlay: model (orange) vs GT (green) on the same observed base — direct comparison
    write_ply(out / f"overlay_model_vs_gt_step{step//1000}k.ply",
              [(obs, BLUE), (gt, GREEN), (model, ORANGE)])


if __name__ == "__main__":
    main()
