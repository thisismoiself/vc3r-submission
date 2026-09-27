#!/usr/bin/env python3
"""Stage 1 full training: deterministic 3D-UNet TSDF completion on cached multi-scene crops.

Trains on SCRREAM scene caches (held-out scene(s) + Replica/NeuralRGBD kept for cross-dataset
val). Checkpoints periodically and is resumable, so it runs unattended under SLURM.
"""
import argparse, glob, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import TSDFCropDataset, VOXEL, BAND, UNKNOWN
from stage1_unet import OcclusionUNet, masked_surface_l1

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(net, ds, n_batches, band):
    net.eval()
    dl = DataLoader(ds, batch_size=2, num_workers=2)
    mae, sign, base, seen = 0.0, 0.0, 0.0, 0
    for b, (inp, gt, mask) in enumerate(dl):
        inp, gt, mask = inp.to(DEV), gt.to(DEV), mask.to(DEV)
        pred = net(inp)
        m = (mask == UNKNOWN) & (gt.abs() < band - VOXEL)
        if m.any():
            mae += (pred - gt).abs()[m].mean().item()
            sign += (pred.sign() == gt.sign())[m].float().mean().item()
            # baseline: partial tsdf channel (input ch0 * band) vs gt
            base += (inp[:, :1] * band - gt).abs()[m].mean().item()
            seen += 1
        if b + 1 >= n_batches:
            break
    net.train()
    n = max(seen, 1)
    return mae / n, sign / n, base / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache")
    ap.add_argument("--val-scenes", nargs="*", default=["scene11", "office4"],
                    help="cache stems held out for validation (cross-dataset)")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--ckpt-every", type=int, default=4000)
    ap.add_argument("--ckpt", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/stage1_scrream.pt")
    args = ap.parse_args()

    caches = sorted(glob.glob(f"{args.cache_dir}/*.npz"))
    val = [c for c in caches if Path(c).stem in args.val_scenes]
    train = [c for c in caches if c not in val]
    print(f"[data] train caches: {[Path(c).stem for c in train]}")
    print(f"[data] val caches:   {[Path(c).stem for c in val]}", flush=True)
    if not train:
        raise SystemExit("no training caches found — run the precompute first")

    train_ds = TSDFCropDataset(train, samples_per_epoch=10 ** 9)      # effectively infinite stream
    val_dss = {Path(c).stem: TSDFCropDataset([c], samples_per_epoch=40, seed=1) for c in val}
    dl = iter(DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True))

    net = OcclusionUNet(in_ch=5, base=args.base, band=BAND).to(DEV)
    print(f"[model] params={sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    surf_band = BAND - VOXEL

    start = 1
    if Path(args.ckpt).exists():                                      # resume
        d = torch.load(args.ckpt, map_location=DEV, weights_only=False)
        net.load_state_dict(d["model"]); opt.load_state_dict(d["opt"]); sched.load_state_dict(d["sched"])
        start = d["step"] + 1
        print(f"[resume] from step {start}", flush=True)

    net.train(); t0 = time.time()
    for step in range(start, args.steps + 1):
        inp, gt, mask = next(dl)
        inp, gt, mask = inp.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True), mask.to(DEV, non_blocking=True)
        with torch.cuda.amp.autocast():
            pred = net(inp); loss = masked_surface_l1(pred, gt, mask, surf_band)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

        if step % 200 == 0 or step == start:
            el = time.time() - t0; it = step - start + 1
            print(f"[train] step {step}/{args.steps} loss={loss.item()*100:.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {el/it:.2f}s/it", flush=True)
        if step % args.eval_every == 0:
            for name, vds in val_dss.items():
                mae, sign, base = evaluate(net, vds, 20, BAND)
                print(f"[eval@{step}] {name}: unk-surf MAE={mae*100:.3f}cm (baseline {base*100:.2f}) "
                      f"sign={sign*100:.1f}%", flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "args": vars(args)}, args.ckpt)
            print(f"[ckpt] step {step} -> {args.ckpt}", flush=True)
    print("[done] Stage 1 training complete.", flush=True)


if __name__ == "__main__":
    main()
