#!/usr/bin/env python3
"""Stage-2 diffusion training (diffusion_training_spec.md): DiffComplete-style DDPM for TSDF
completion. Trains on cached 3D-FRONT room crops; SCRREAM is eval-only.

Modes:
  --overfit CACHE.npz   Verification gate (build order step 3): fit ONE crop and DDIM-sample it
                        back. If the sampled TSDF reproduces the crop's GT on unknown voxels, the
                        whole machinery — cosine schedule, q_sample, epsilon loss, control-branch
                        injection, DDIM sampler — is correct. Do this before the full run.
  (default)             Full Stage-A training: stream frontier-biased crops from --cache-dir,
                        AdamW + cosine LR + warmup, bf16, EMA(0.9999), resumable checkpoints.

x0 (the thing we generate) = GT TSDF normalized to [-1,1] by dividing by the truncation band.
cond (5ch) = normalized partial TSDF + observed/free/unknown one-hot + confidence. Loss is L1 on
the predicted NOISE, masked to unknown voxels, surface-weighted 5x (weight computed from x0).
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import TSDFCropDataset, CoarseRoomDataset, VOXEL, BAND, UNKNOWN
from diffcomplete_ddpm import DiffCompleteUNet, Diffusion, EMA

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SURF_BAND_M = BAND - VOXEL                                       # near-surface distance threshold (m)


def unpack(batch):
    """Dataset emits (inp[5], gt[1], mask[1]) with gt a SIGNED TSDF in metres.

    Following DiffComplete, the GENERATED target x0 is an UNSIGNED TUDF (derived here as |TSDF|),
    while the CONDITIONING stays a signed TSDF (cond channel 0). The unsigned target removes the
    ill-posed sign field that 3D-FRONT's fully-enclosed rooms create (interior air reads as
    'inside' the wall shell -> negative). UDF is normalized to [-1,1]: surface(0m)->-1, far(band)->+1.
    Returns x0, cond, mask, and a near-surface bool mask for the loss weighting."""
    inp, gt, mask = batch
    gt = gt.to(DEV)
    udf = gt.abs().clamp(0, BAND)                                # unsigned truncated distance (m)
    x0 = (2 * udf / BAND - 1).clamp(-1, 1)                       # normalized UDF in [-1,1]
    near = udf < SURF_BAND_M
    return x0, inp.to(DEV), mask.to(DEV), near


def _x0_to_udf_m(x):
    """Normalized-UDF [-1,1] -> unsigned distance in metres."""
    return (x.clamp(-1, 1) + 1) * 0.5 * BAND


@torch.no_grad()
def sample_and_score(model, diff, x0, cond, mask, near, steps=50, guidance=1.0, replace=False):
    """DDIM-sample a completion and score it against the GT UDF on unknown voxels:
    - MAE: mean |udf_pred - udf_gt| over near-surface unknown voxels (cm).
    - surf-recall: of the GT surface voxels (udf<1 voxel), the fraction the model also places a
      surface at (udf_pred < 2 voxels). Replaces the sign metric, which is meaningless for a UDF."""
    x_known = (2 * cond[:, :1].abs() - 1)                        # partial TSDF -> normalized UDF space
    pred = diff.ddim_sample(model, cond, mask=mask, x_known=x_known, steps=steps,
                            guidance=guidance, replace=replace)
    udf_pred = _x0_to_udf_m(pred); udf_gt = _x0_to_udf_m(x0)
    m = (mask == UNKNOWN) & near
    mae = (udf_pred - udf_gt).abs()[m].mean().item() * 100 if m.any() else float("nan")
    surf = (mask == UNKNOWN) & (udf_gt < VOXEL)
    rec = (udf_pred[surf] < 2 * VOXEL).float().mean().item() * 100 if surf.any() else float("nan")
    return mae, rec


def overfit(args):
    """Fit a single crop and check the sampler reproduces it."""
    ds = TSDFCropDataset([args.overfit], samples_per_epoch=1, seed=0)
    x0, cond, mask, near = unpack(tuple(t[None] for t in ds[0]))
    print(f"[overfit] crop unknown={int((mask==UNKNOWN).sum())}  "
          f"target(unk near surf)={int(((mask==UNKNOWN)&near).sum())}", flush=True)

    model = DiffCompleteUNet(use_ckpt=True).to(DEV)
    diff = Diffusion(T=1000, device=DEV)
    ema = EMA(model, decay=0.999)                                 # sample from EMA, like real training
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    # bf16 on Ampere+, else fp16 (Turing, e.g. RTX 5000/6000) with a grad scaler
    amp_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dt == torch.float16))
    model.train(); t0 = time.time()
    for step in range(1, args.steps + 1):
        with torch.autocast("cuda", dtype=amp_dt):
            loss = diff.p_losses(model, x0, cond, mask, near)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ema.update(model)
        if step % 100 == 0 or step == 1:
            print(f"  step {step}/{args.steps} loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
        if step % args.sample_every == 0:
            em = DiffCompleteUNet(use_ckpt=False).to(DEV); ema.copy_to(em); em.eval()
            with torch.autocast("cuda", dtype=amp_dt):
                mae, rec = sample_and_score(em, diff, x0, cond, mask, near, steps=50)
                mae_r, _ = sample_and_score(em, diff, x0, cond, mask, near, steps=50, replace=True)
            print(f"  [sample@{step}] EMA unk-surf UDF-MAE={mae:.3f}cm (replace {mae_r:.3f}) "
                  f"surf-recall={rec:.1f}%", flush=True)
            del em; torch.cuda.empty_cache()
    # dump the final EMA completion for visualization (partial in, GT complete, model completion)
    if args.viz_out:
        em = DiffCompleteUNet(use_ckpt=False).to(DEV); ema.copy_to(em); em.eval()
        with torch.autocast("cuda", dtype=amp_dt):
            pred = diff.ddim_sample(em, cond, mask=mask, x_known=(2 * cond[:, :1].abs() - 1),
                                    steps=50, replace=True)
        udf_pred = ((pred.float() + 1) * 0.5 * BAND).clamp(0, BAND)[0, 0].cpu().numpy()
        udf_gt = ((x0.float() + 1) * 0.5 * BAND)[0, 0].cpu().numpy()
        part = (cond[:, 0].float() * BAND)[0].cpu().numpy()          # signed partial TSDF (m)
        Path(args.viz_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.viz_out, udf_pred=udf_pred.astype(np.float16),
                            udf_gt=udf_gt.astype(np.float16), partial_tsdf=part.astype(np.float16),
                            mask=mask[0, 0].cpu().numpy().astype(np.int8), voxel=VOXEL, band=BAND)
        print(f"[viz] wrote completion arrays -> {args.viz_out}", flush=True)
    print("[verdict] PASS if unk-surf UDF-MAE -> ~1 voxel (<=2cm) and surf-recall -> ~100%.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--max-scenes", type=int, default=1500, help="cap caches loaded into RAM")
    ap.add_argument("--overfit", default=None, help="single cache npz -> run the overfit gate")
    ap.add_argument("--viz-out", default=None, help="overfit: dump final completion arrays here (npz)")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=0, help="0=load all rooms; >0 STREAM this many resident, cycling through every cached room (use all 12k)")
    ap.add_argument("--refresh-every", type=int, default=150, help="streaming: swap one room off disk every N samples")
    ap.add_argument("--cond-drop", type=float, default=0.0, help="prob of dropping cond (CFG); 0=off")
    ap.add_argument("--init-from", default=None, help="warm-start weights (Stage-B fine-tune); fresh optimizer")
    ap.add_argument("--surf-w", type=float, default=5.0, help="near-surface loss weight; lower (~2) curbs over-prediction")
    ap.add_argument("--frontier-dilate", type=int, default=0, help="voxels: restrict loss to unknown within this dist of observed surface (0=all unknown)")
    ap.add_argument("--band", type=float, default=0.0, help="override truncation band (m) to match the cache, e.g. 0.06 for the v2 caches; 0=use default 0.10")
    ap.add_argument("--da3-sim", action="store_true", help="corrupt clean partials to DA3-like (jitter+surface holes) so 3D-FRONT teaches DA3-hole-filling")
    ap.add_argument("--coarse", action="store_true", help="cascade coarse stage: whole-room (no crop) completion at coarse resolution")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--ckpt", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/diffusion_stageA.pt")
    args = ap.parse_args()

    if args.band > 0:                                           # match the cache's truncation band
        global BAND, SURF_BAND_M
        BAND = args.band; SURF_BAND_M = BAND - VOXEL
        print(f"[band] overriding truncation band -> {BAND} m (SURF_BAND_M={SURF_BAND_M})", flush=True)

    if args.overfit:
        args.steps = args.steps if args.steps < 100000 else 3000
        overfit(args); return

    caches = sorted(glob.glob(f"{args.cache_dir}/*.npz"))
    if not caches:
        raise SystemExit(f"no caches in {args.cache_dir} — run front3d_assembler first")
    rng = np.random.default_rng(0); rng.shuffle(caches)
    if not args.pool_size:                                       # streaming uses ALL cached rooms
        caches = caches[:args.max_scenes]
    n_val = int(len(caches) * args.val_frac)                    # 0 allowed (few-scene fine-tune: use all for train)
    val, train = caches[:n_val], caches[n_val:]
    print(f"[data] {len(train)} train / {len(val)} val room caches", flush=True)

    DS = CoarseRoomDataset if args.coarse else TSDFCropDataset
    kw = {} if args.coarse else {"da3_sim": args.da3_sim}
    train_kw = dict(kw)
    if not args.coarse and args.pool_size:                       # STREAM the train set; val stays small full-load
        train_kw.update(pool_size=args.pool_size, refresh_every=args.refresh_every)
    train_ds = DS(train, band=BAND, samples_per_epoch=10 ** 9, **train_kw)
    val_ds = DS(val, band=BAND, samples_per_epoch=64, seed=1)
    dl = iter(DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=True, persistent_workers=True))
    val_dl = DataLoader(val_ds, batch_size=2, num_workers=2)

    model = DiffCompleteUNet(use_ckpt=True).to(DEV)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    diff = Diffusion(T=1000, device=DEV)
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # bf16 on Ampere+ (all the GPUs the sbatch constrains to); fp16 + scaler otherwise, so the job
    # never crashes on whatever card SLURM assigns.
    amp_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dt == torch.float16))
    print(f"[amp] dtype={amp_dt}", flush=True)

    def lr_at(step):                                            # linear warmup then cosine decay
        if step < args.warmup:
            return step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    start = 1
    if Path(args.ckpt).exists():
        d = torch.load(args.ckpt, map_location=DEV, weights_only=False)
        model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"]); sched.load_state_dict(d["sched"])
        ema.shadow = {k: v.to(DEV) for k, v in d["ema"].items()}; start = d["step"] + 1
        print(f"[resume] from step {start}", flush=True)
    elif args.init_from and Path(args.init_from).exists():
        # Stage-B WARM START: load best Stage-A weights (EMA quality) as init, FRESH optimizer/schedule
        # (this is fine-tuning on new data, not resuming). EMA re-snapshots from the loaded model.
        d = torch.load(args.init_from, map_location=DEV, weights_only=False)
        init_w = d.get("ema", d.get("model"))
        model.load_state_dict(init_w)
        ema = EMA(model, decay=0.9999)
        print(f"[init] warm-start from {args.init_from} (Stage-A step {d.get('step')}, EMA weights); "
              f"fresh optimizer, surf_w={args.surf_w}", flush=True)

    model.train(); t0 = time.time()
    for step in range(start, args.steps + 1):
        x0, cond, mask, near = unpack(next(dl))
        cd = (torch.rand(x0.shape[0], device=DEV) < args.cond_drop) if args.cond_drop > 0 else None
        with torch.autocast("cuda", dtype=amp_dt):
            loss = diff.p_losses(model, x0, cond, mask, near, cond_drop=cd, surf_w=args.surf_w,
                                 frontier_dilate=args.frontier_dilate)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); ema.update(model)

        if step % 100 == 0 or step == start:
            el = time.time() - t0; it = step - start + 1
            print(f"[train] step {step}/{args.steps} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {el/it:.2f}s/it", flush=True)
        if val and step % args.eval_every == 0:
            ema_model = DiffCompleteUNet(use_ckpt=False).to(DEV); ema.copy_to(ema_model); ema_model.eval()
            maes, recs = [], []
            for b in val_dl:
                x0, cond, mask, near = unpack(b)
                mae, rec = sample_and_score(ema_model, diff, x0, cond, mask, near, steps=50)
                if mae == mae:
                    maes.append(mae); recs.append(rec)
            print(f"[eval@{step}] EMA val unk-surf UDF-MAE={np.mean(maes):.3f}cm "
                  f"surf-recall={np.mean(recs):.1f}% (n={len(maes)})", flush=True)
            del ema_model; torch.cuda.empty_cache()
        if step % args.ckpt_every == 0 or step == args.steps:
            Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.shadow, "step": step, "args": vars(args)}, args.ckpt)
            print(f"[ckpt] step {step} -> {args.ckpt}", flush=True)
    print("[done] Stage A complete.", flush=True)


if __name__ == "__main__":
    main()
