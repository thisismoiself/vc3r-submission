#!/usr/bin/env python3
"""Diagnostic: is the step-84k regression in the EMA weights, the raw weights, or both? Evaluate the
same held-out rooms with d['ema'] vs d['model'] and compare F / precision / hallucination."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from train_diffusion import DEV
from diffcomplete_ddpm import DiffCompleteUNet, Diffusion
from progress_strip import val_rooms, fixed_crop, base_geometry, sample_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/snapshots/stageA_step84k_degraded.pt")
    ap.add_argument("--room-idxs", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    d = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    diff = Diffusion(T=1000, device=DEV)
    vr = val_rooms("/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    crops = {}
    for r in args.room_idxs:
        x0, cond, mask, near = fixed_crop(vr[r], seed=0)
        crops[r] = (x0, cond, mask, near, base_geometry(x0, cond, mask, near))

    for tag in ["ema", "model"]:
        m = DiffCompleteUNet(use_ckpt=False).to(DEV); m.load_state_dict(d[tag]); m.eval()
        print(f"\n=== weights = d['{tag}'] (step {d['step']}) ===", flush=True)
        for r in args.room_idxs:
            x0, cond, mask, near, base = crops[r]
            _, mt, _ = sample_frame(m, diff, x0, cond, mask, near, base, steps=50)
            print(f"  room{r}: F={mt['F']:.1f}  recall={mt['recall']:.1f}%  prec={mt['prec']:.1f}%  "
                  f"halluc={mt['halluc']:.1f}%", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
