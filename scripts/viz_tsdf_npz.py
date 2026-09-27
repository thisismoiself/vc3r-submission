#!/usr/bin/env python3
"""Visualize a cached full-scene TSDF grid (from tsdf_dataset / front3d_assembler).
Renders orthogonal slices of GT TSDF, partial TSDF, and the observed/free/unknown mask, plus a
projected view of where the training target lives (unknown voxels near the GT surface).
This is the spec's mandatory "look at the masks and both TSDFs before training" gate."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OBSERVED, FREE, UNKNOWN = 0, 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = np.load(args.npz)
    gt = d["gt_tsdf"].astype(np.float32); pt = d["partial_tsdf"].astype(np.float32)
    mask = d["mask"]; voxel = float(d["voxel"]); band = float(np.abs(gt).max())
    dims = gt.shape
    nun = (mask == UNKNOWN).sum()
    tgt = (mask == UNKNOWN) & (np.abs(gt) < band - voxel)
    print(f"dims={dims} voxel={voxel*100:.0f}cm band={band*100:.0f}cm  "
          f"unknown={100*nun/mask.size:.0f}%  target(unk near surf)={int(tgt.sum()):,}  "
          f"gt[{gt.min():.3f},{gt.max():.3f}] part[{pt.min():.3f},{pt.max():.3f}]")

    cmap_mask = mcolors.ListedColormap(["#22e022", "#3399ff", "#ee2222"])  # obs / free / unknown
    # take 3 horizontal (constant-y, the up axis) slices through the furnished volume
    ys = [dims[1] // 4, dims[1] // 2, 3 * dims[1] // 4]
    fig, ax = plt.subplots(4, 3, figsize=(13, 15))
    for col, y in enumerate(ys):
        ax[0, col].imshow(gt[:, y, :], cmap="coolwarm", vmin=-band, vmax=band)
        ax[0, col].set_title(f"GT TSDF  y={y}")
        ax[1, col].imshow(pt[:, y, :], cmap="coolwarm", vmin=-band, vmax=band)
        ax[1, col].set_title(f"partial TSDF  y={y}")
        ax[2, col].imshow(mask[:, y, :], cmap=cmap_mask, vmin=0, vmax=2)
        ax[2, col].set_title(f"mask (grn=obs blu=free red=unk)  y={y}")
        # target overlay: GT surface (black) with unknown-target voxels (orange) on top
        surf = (np.abs(gt[:, y, :]) < band - voxel).astype(float)
        tg = tgt[:, y, :].astype(float)
        rgb = np.stack([surf * 0.2 + tg, surf * 0.2 + tg * 0.5, surf * 0.2], -1)
        ax[3, col].imshow(np.clip(rgb, 0, 1)); ax[3, col].set_title(f"surface(gray)+target(orange) y={y}")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    out = args.out or (Path(args.npz).with_suffix(".viz.png"))
    plt.savefig(out, dpi=100); plt.close()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
