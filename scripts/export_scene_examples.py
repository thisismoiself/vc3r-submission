#!/usr/bin/env python3
"""Export ScanNet++ cached scenes (the fine-tuning data itself) as coloured PLYs for inspection:
grey = complete GT surface, blue = observed (partial scan), red = occluded region to complete.
Lets us eyeball whether the carving/GT is sane on real data (CPU-only, no model involved)."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from export_ply import write_ply

OBSERVED, FREE, UNKNOWN = 0, 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=300000)
    args = ap.parse_args()
    d = np.load(args.npz); v = float(d["voxel"]); m = d["mask"]
    gt = np.abs(d["gt_tsdf"].astype(np.float32)); pt = np.abs(d["partial_tsdf"].astype(np.float32))
    surf = 0.7 * v
    gt_s = gt < surf                                   # complete GT surface
    obs = (m == OBSERVED) & (pt < surf)                # observed partial surface
    occ = (m == UNKNOWN) & gt_s                        # occluded GT surface (target to complete)
    rng = np.random.default_rng(0)
    def P(g):
        idx = np.argwhere(g).astype(np.float32)
        if len(idx) > args.cap: idx = idx[rng.choice(len(idx), args.cap, replace=False)]
        return idx * v
    name = Path(args.npz).stem
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    GREY, BLUE, RED = (150, 150, 150), (60, 130, 210), (225, 50, 40)
    write_ply(out / f"{name}_GT_complete.ply", [(P(gt_s), GREY)])
    write_ply(out / f"{name}_observed.ply", [(P(obs), BLUE)])
    write_ply(out / f"{name}_observed+occluded.ply", [(P(obs), BLUE), (P(occ), RED)])
    print(f"{name}: obs={int(obs.sum())} occ={int(occ.sum())} gt_surf={int(gt_s.sum())} "
          f"unknown={100*(m==UNKNOWN).mean():.0f}%", flush=True)


if __name__ == "__main__":
    main()
