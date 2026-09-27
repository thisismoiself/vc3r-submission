#!/usr/bin/env python3
"""Evaluate a checkpoint on held-out ScanNet++ scenes with the precision-aware metrics (F / recall /
precision / hallucination) — the honest metric, not the MAE/recall the built-in eval uses. Used to
compare: (a) stageA_BEST zero-shot on real scans, vs (b) the Stage-B fine-tuned model.

Takes explicit cache npz paths (no front3d split logic), so it works on any TSDF cache dir."""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from train_diffusion import DEV
from diffcomplete_ddpm import DiffCompleteUNet, Diffusion
from progress_strip import fixed_crop, base_geometry, sample_frame, load_ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_scannetpp")
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--which", choices=["last", "first"], default="last", help="held-out slice of the cache")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    caches = sorted(glob.glob(f"{args.cache_dir}/*.npz"))
    sel = caches[-args.n_scenes:] if args.which == "last" else caches[:args.n_scenes]
    d = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    model = DiffCompleteUNet(use_ckpt=False).to(DEV)
    model.load_state_dict(d[args.weights]); model.eval()
    diff = Diffusion(T=1000, device=DEV)
    print(f"[eval {args.tag}] ckpt={Path(args.ckpt).name} weights={args.weights} step={d.get('step')} "
          f"on {len(sel)} held-out scenes", flush=True)

    Fs, Ps, Rs, Hs = [], [], [], []
    for c in sel:
        name = Path(c).stem[:32]
        try:
            x0, cond, mask, near = fixed_crop(c, seed=0)
            base = base_geometry(x0, cond, mask, near)
            _, mt, _ = sample_frame(model, diff, x0, cond, mask, near, base, args.steps)
        except Exception as e:
            print(f"  {name}: FAILED {e}", flush=True); continue
        Fs.append(mt["F"]); Ps.append(mt["prec"]); Rs.append(mt["recall"]); Hs.append(mt["halluc"])
        print(f"  {name}: F={mt['F']:.1f}  recall={mt['recall']:.1f}%  prec={mt['prec']:.1f}%  "
              f"halluc={mt['halluc']:.1f}%", flush=True)
    if Fs:
        print(f"[summary {args.tag}] n={len(Fs)}  F={np.mean(Fs):.1f}  recall={np.mean(Rs):.1f}%  "
              f"prec={np.mean(Ps):.1f}%  halluc={np.mean(Hs):.1f}%", flush=True)


if __name__ == "__main__":
    main()
