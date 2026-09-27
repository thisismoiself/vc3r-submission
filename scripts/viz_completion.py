#!/usr/bin/env python3
"""Render 'what's working' figures as 3D point-cloud surfaces (voxel centres near the zero-level),
headless via matplotlib. Two modes, auto-detected from the npz keys:

  * COMPLETION crop (from train_diffusion --viz-out): keys udf_pred, udf_gt, partial_tsdf, mask.
    Renders 3 panels from one viewpoint: partial INPUT | model COMPLETION | GT complete.
    The model-generated geometry (unknown voxels the model fills) is drawn in orange, so you can
    see the occluded region get filled and compare to the green GT.

  * ROOM cache (from front3d_assembler): keys gt_tsdf, partial_tsdf, mask.
    Renders the complete assembled surface, the partial (observed) surface, and the occluded
    (unknown) region — i.e. the data pipeline itself.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBSERVED, FREE, UNKNOWN = 0, 1, 2


def _pts(bool_grid, voxel, rng, cap=45000):
    idx = np.argwhere(bool_grid)
    if len(idx) > cap:
        idx = idx[rng.choice(len(idx), cap, replace=False)]
    return idx * voxel


def _scatter(ax, groups, elev=18, azim=-60, title=""):
    for pts, c, s, a in groups:
        if len(pts):
            ax.scatter(pts[:, 0], -pts[:, 2], pts[:, 1], c=c, s=s, alpha=a, linewidths=0, marker=".")  # -Z keeps right-handed (no mirror), Y up
    ax.set_title(title, fontsize=13)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.6)); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--elev", type=float, default=20)
    ap.add_argument("--azim", type=float, default=-70)
    args = ap.parse_args()
    d = np.load(args.npz)
    voxel = float(d["voxel"]); rng = np.random.default_rng(0)
    mask = d["mask"]
    surf = 0.7 * voxel                                             # 'on a surface' = within ~1 voxel

    if "udf_pred" in d:                                            # ---- completion crop ----
        udf_pred = d["udf_pred"].astype(np.float32); udf_gt = d["udf_gt"].astype(np.float32)
        part = np.abs(d["partial_tsdf"].astype(np.float32))
        obs = (mask == OBSERVED) & (part < surf)                  # observed input surface
        gt_new = (mask == UNKNOWN) & (udf_gt < surf)              # GT occluded surface
        # For the model, show its MOST-CONFIDENT surface at the SAME density as GT (rank unknown
        # voxels by predicted distance, take the closest N). This reflects the true recall/precision
        # instead of a threshold artifact — a soft UDF thresholded naïvely looks like a cloud;
        # a real mesh comes from marching-cubes on the zero level-set.
        unk = (mask == UNKNOWN)
        n_gt = int(gt_new.sum())
        pred_new = np.zeros_like(unk)
        if n_gt and unk.sum() > n_gt:
            uv = udf_pred[unk]
            thr = np.partition(uv, n_gt - 1)[n_gt - 1]
            pred_new = unk & (udf_pred <= thr)
        o = _pts(obs, voxel, rng); pn = _pts(pred_new, voxel, rng); gn = _pts(gt_new, voxel, rng)
        fig = plt.figure(figsize=(16, 6))
        titles = ["INPUT: partial DA3-style scan (holes)",
                  "MODEL: DiffComplete fills occluded voxels (orange)",
                  "GROUND TRUTH: complete geometry (green)"]
        panels = [[(o, "#2f7fd0", 3, .5)],
                  [(o, "#2f7fd0", 3, .4), (pn, "#ff7a1a", 5, .8)],
                  [(o, "#2f7fd0", 3, .4), (gn, "#1eb84f", 5, .8)]]
        for i, (t, g) in enumerate(zip(titles, panels)):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            _scatter(ax, g, args.elev, args.azim, t)
        info = f"128^3 crop @ {voxel*100:.0f}cm | model-filled voxels orange vs GT green"
    else:                                                          # ---- room / data pipeline ----
        gt = np.abs(d["gt_tsdf"].astype(np.float32)); part = np.abs(d["partial_tsdf"].astype(np.float32))
        gt_surf = gt < surf; obs = (mask == OBSERVED) & (part < surf)
        occl = (mask == UNKNOWN) & gt_surf                        # occluded GT surface (to be filled)
        g = _pts(gt_surf, voxel, rng); o = _pts(obs, voxel, rng); oc = _pts(occl, voxel, rng)
        fig = plt.figure(figsize=(16, 6))
        panels = [("COMPLETE assembled room (GT)", [(g, "#888888", 2, .5)]),
                  ("OBSERVED (simulated partial scan)", [(o, "#2f7fd0", 2, .5)]),
                  ("OCCLUDED region to complete (red)", [(o, "#2f7fd0", 2, .3), (oc, "#e23", 3, .7)])]
        for i, (t, gr) in enumerate(panels):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            _scatter(ax, gr, args.elev, args.azim, t)
        info = f"3D-FRONT room, full grid @ {voxel*100:.0f}cm"

    fig.suptitle(info, fontsize=11, y=0.04)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=115, bbox_inches="tight"); plt.close()
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
