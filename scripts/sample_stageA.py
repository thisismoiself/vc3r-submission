#!/usr/bin/env python3
"""Sample completions from a trained Stage-A checkpoint (EMA weights) on a few rooms and dump
per-crop completion npzs for visualization. Rooms are chosen from across the cache; since the
step-16k checkpoint was trained on only its startup snapshot, rooms outside that are effectively
held out. Runs the real DDPM + DDIM sampler — this is the current model, not the overfit."""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import TSDFCropDataset, VOXEL, BAND, UNKNOWN
from train_diffusion import unpack, DEV
from diffcomplete_ddpm import DiffCompleteUNet, Diffusion


def pick_crop(room_npz, tries=6, seed=0):
    """Return the frontier crop (of a few tries) with the most unknown near-surface target voxels."""
    ds = TSDFCropDataset([room_npz], samples_per_epoch=tries, seed=seed)
    best, best_n = None, -1
    for i in range(tries):
        inp, gt, msk = ds[i]
        x0, cond, mask, near = unpack(tuple(t[None] for t in (inp, gt, msk)))
        n = int(((mask == UNKNOWN) & near).sum())
        if n > best_n:
            best_n, best = n, (x0, cond, mask, near)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    ap.add_argument("--rooms", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    d = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    step = d.get("step", 0)
    model = DiffCompleteUNet(use_ckpt=False).to(DEV)
    model.load_state_dict(d["ema"]); model.eval()               # EMA weights
    diff = Diffusion(T=1000, device=DEV)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[ckpt] step {step}, EMA loaded, amp={amp}", flush=True)

    caches = sorted(glob.glob(f"{args.cache_dir}/*.npz"))
    # spread choices across the cache (skip the first few that were likely in the early train set)
    idxs = np.linspace(len(caches) // 3, len(caches) - 1, args.rooms).round().astype(int)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for j, ci in enumerate(idxs):
        room = caches[ci]; name = Path(room).stem[:28]
        x0, cond, mask, near = pick_crop(room, seed=j)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            pred = diff.ddim_sample(model, cond, mask=mask, x_known=(2 * cond[:, :1].abs() - 1),
                                    steps=args.steps, replace=True)
        udf_pred = ((pred.float() + 1) * 0.5 * BAND).clamp(0, BAND)[0, 0].cpu().numpy()
        udf_gt = ((x0.float() + 1) * 0.5 * BAND)[0, 0].cpu().numpy()
        part = (cond[:, 0].float() * BAND)[0].cpu().numpy()
        m = (mask == UNKNOWN) & near
        mae = (torch.tensor(udf_pred) - torch.tensor(udf_gt)).abs()[m[0, 0].cpu()].mean().item() * 100
        np.savez_compressed(out / f"room{j}_{name}.npz", udf_pred=udf_pred.astype(np.float16),
                            udf_gt=udf_gt.astype(np.float16), partial_tsdf=part.astype(np.float16),
                            mask=mask[0, 0].cpu().numpy().astype(np.int8), voxel=VOXEL, band=BAND)
        print(f"[room {j}] {name}  unk-surf UDF-MAE={mae:.2f}cm -> room{j}_{name}.npz", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
