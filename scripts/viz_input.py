#!/usr/bin/env python3
"""Visualise EXACTLY what the DiffComplete network is given as input for one room's crop — no RGB,
no depth images (there are none): the input is a volumetric partial TSDF + an observed/free/unknown
mask. Regenerates the deterministic hero crop (fixed_crop seed 0) and renders:

  (A) 3D: the OBSERVED partial surface (blue, what's given) vs the UNKNOWN region (grey, to fill).
  (B) 2D orthographic slices — the literal volumetric 'images/masks' the 3D conv sees:
       left  = partial TSDF (signed distance, diverging colormap)
       right = mask (observed / free / unknown)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from progress_strip import val_rooms, fixed_crop
from tsdf_dataset import VOXEL, BAND

OBSERVED, FREE, UNKNOWN = 0, 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    ap.add_argument("--room-idx", type=int, default=2)
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/experiments/overfit_8frames/eval_rooms/input_room2")
    args = ap.parse_args()

    vr = val_rooms(args.cache_dir); room = vr[args.room_idx]; name = Path(room).stem[:36]
    x0, cond, mask, near = fixed_crop(room, seed=0)
    part = (cond[0, 0].float() * BAND).cpu().numpy()    # ch0: partial signed TSDF (m)
    m = mask[0, 0].cpu().numpy()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    surf = 0.7 * VOXEL
    print(f"[room {args.room_idx}] {name}  crop {m.shape}  "
          f"observed={100*(m==OBSERVED).mean():.0f}% free={100*(m==FREE).mean():.0f}% "
          f"unknown={100*(m==UNKNOWN).mean():.0f}%", flush=True)

    # ---- (A) 3D input point cloud: observed surface vs unknown region ----
    rng = np.random.default_rng(0)
    def pts(g, cap):
        idx = np.argwhere(g)
        if len(idx) > cap: idx = idx[rng.choice(len(idx), cap, replace=False)]
        return idx * VOXEL
    obs_p = pts((m == OBSERVED) & (np.abs(part) < surf), 45000)      # given surface
    unk_p = pts(m == UNKNOWN, 30000)                                  # hidden region to complete
    fig = plt.figure(figsize=(16, 6))
    for i, (title, groups) in enumerate([
        ("INPUT surface the model SEES\n(observed partial TSDF)", [(obs_p, "#2f7fd0", 4, .6)]),
        ("UNKNOWN region to COMPLETE\n(grey = occluded/unseen)", [(unk_p, "#999999", 2, .25),
                                                                   (obs_p, "#2f7fd0", 4, .6)]),
        ("what's given vs what's hidden", [(unk_p, "#c62", 2, .18), (obs_p, "#2f7fd0", 4, .7)])]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for p, c, s, a in groups:
            if len(p): ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=c, s=s, alpha=a, linewidths=0, marker=".")
        ax.set_title(title, fontsize=12); ax.view_init(elev=20, azim=-70)
        ax.set_box_aspect((1, 1, 0.6)); ax.axis("off")
    fig.suptitle(f"MODEL INPUT (3D volume, not images) — {name}  @ {VOXEL*100:.0f}cm voxels", y=0.04, fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(out / "input_3d.png", dpi=115, bbox_inches="tight"); plt.close()
    print(f"-> {out}/input_3d.png", flush=True)

    # ---- (B) 2D orthographic slices: partial TSDF + mask (the literal input tensors) ----
    D = m.shape[1]                                                    # slice along the up (y) axis
    zs = [int(D * f) for f in (0.30, 0.45, 0.60)]                     # a few informative heights
    mask_cmap = ListedColormap(["#2f7fd0", "#eeeeee", "#e2b23a"])     # observed / free / unknown
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], mask_cmap.N)
    fig, axes = plt.subplots(len(zs), 2, figsize=(9, 4 * len(zs)))
    for r, z in enumerate(zs):
        pt_sl = part[:, z, :].T; mk_sl = m[:, z, :].T
        a0 = axes[r, 0].imshow(pt_sl, cmap="coolwarm", vmin=-BAND, vmax=BAND, origin="lower")
        axes[r, 0].set_title(f"partial TSDF  (y-slice {z}/{D})", fontsize=10); axes[r, 0].axis("off")
        fig.colorbar(a0, ax=axes[r, 0], fraction=0.046, label="signed dist (m)")
        axes[r, 1].imshow(mk_sl, cmap=mask_cmap, norm=norm, origin="lower")
        axes[r, 1].set_title("mask  (blue=observed  grey=free  gold=unknown)", fontsize=10); axes[r, 1].axis("off")
    fig.suptitle(f"MODEL INPUT tensors, horizontal slices — {name}", y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(out / "input_slices.png", dpi=115, bbox_inches="tight"); plt.close()
    print(f"-> {out}/input_slices.png", flush=True)


if __name__ == "__main__":
    main()
