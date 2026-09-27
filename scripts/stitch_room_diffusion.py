#!/usr/bin/env python3
"""Full-room completion for the DiffComplete TSDF model by TILED diffusion.

The U-Net only sees a 128^3 (2.56 m) cube, but a room is bigger. We slide OVERLAPPING 128^3
windows across the cached full-room grid, run the diffusion (with RePaint replace) on each, and
cross-fade-blend the per-window predictions into one full-room UDF with a Hann feather weight
(1 at a window's centre, ->0 at its edges) so tile boundaries leave no seam. Then extract the
completed surface, score it against the WHOLE-room GT with the same precision-aware F as the
per-crop eval, and dump coloured PLYs.

NOTE: run on a band=0.10 m cache (matches the model's training band) — 3D-FRONT holdout or Replica.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_dilation, label

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from diffcomplete_ddpm import Diffusion, OBSERVED, FREE, UNKNOWN
from progress_strip import load_ema
from export_ply import write_ply

DEV = torch.device("cuda")


def hann3d(n):
    w = np.hanning(n + 2)[1:-1].astype(np.float32)         # drop the zero endpoints
    w = np.clip(w, 1e-3, None)
    return w[:, None, None] * w[None, :, None] * w[None, None, :]


def positions(dim, win, stride):
    if dim <= win:
        return [0]
    ps = list(range(0, dim - win + 1, stride))
    if ps[-1] != dim - win:
        ps.append(dim - win)
    return ps


def pad_to(a, win, fill):
    out = np.full((win, win, win), fill, a.dtype)
    out[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)      # 64 = 50% overlap
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--frontier-mask", type=int, default=0, help="voxels: restrict prediction to unknown within this dist of observed surface (kills outside-room fill); 0=off")
    ap.add_argument("--cc-filter", action="store_true", help="keep only predicted surface connected to observed structure (drops floating blobs)")
    args = ap.parse_args()

    d = np.load(args.npz)
    pt = d["partial_tsdf"].astype(np.float32)              # signed partial TSDF, metres
    mask = d["mask"].astype(np.int8)
    gt = np.abs(d["gt_tsdf"].astype(np.float32))
    voxel = float(d["voxel"]); band = float(d["band"]) if "band" in d.files else 0.10
    surf = 0.7 * voxel
    D = pt.shape; win = args.win
    if abs(band - 0.10) > 1e-3:
        print(f"[warn] npz band={band} != model training band 0.10 — conditioning will be mis-scaled", flush=True)

    model, step = load_ema(args.ckpt)
    diff = Diffusion(T=1000, device=DEV)
    W = hann3d(win)
    acc = np.zeros(D, np.float32); wacc = np.zeros(D, np.float32)
    xs, ys, zs = (positions(D[i], win, args.stride) for i in range(3))
    xs, ys, zs = list(xs), list(ys), list(zs)
    nwin = len(xs) * len(ys) * len(zs); k = 0
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[room] {Path(args.npz).stem}  grid={D}  band={band}  windows={nwin} "
          f"(win={win} stride={args.stride})  ckpt step={step}", flush=True)

    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                a1, b1, c1 = min(x0 + win, D[0]), min(y0 + win, D[1]), min(z0 + win, D[2])
                vx, vy, vz = a1 - x0, b1 - y0, c1 - z0
                pt_w = pad_to(pt[x0:a1, y0:b1, z0:c1], win, band)
                mk_w = pad_to(mask[x0:a1, y0:b1, z0:c1], win, UNKNOWN)
                pt_n = torch.from_numpy(pt_w / band).float()
                oh = torch.nn.functional.one_hot(torch.from_numpy(mk_w.astype(np.int64)), 3).permute(3, 0, 1, 2).float()
                cond = torch.cat([pt_n[None], oh, torch.ones(1, win, win, win)], 0)[None].to(DEV)
                mk_t = torch.from_numpy(mk_w.astype(np.int64))[None, None].to(DEV)
                with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                    pred = diff.ddim_sample(model, cond, mask=mk_t, x_known=(2 * cond[:, :1].abs() - 1),
                                            steps=args.steps, replace=True)
                udf = ((pred.float() + 1) * 0.5 * band).clamp(0, band)[0, 0].cpu().numpy()
                acc[x0:a1, y0:b1, z0:c1] += (udf * W)[:vx, :vy, :vz]
                wacc[x0:a1, y0:b1, z0:c1] += W[:vx, :vy, :vz]
                k += 1
                print(f"[win {k}/{nwin}] @({x0},{y0},{z0})", flush=True)

    udf_pred = acc / np.clip(wacc, 1e-6, None)

    # --- whole-room precision-aware scoring (same definitions as the per-crop eval) ---
    obs = (mask == OBSERVED) & (np.abs(pt) < surf)
    udf_gt = np.clip(gt, 0, band)
    gt_new = (mask == UNKNOWN) & (udf_gt < surf)
    true_surf = obs | gt_new
    unk = (mask == UNKNOWN); n_gt = int(gt_new.sum())
    cand = unk
    if args.frontier_mask > 0:                             # restrict to unknown near observed surface
        near_obs = binary_dilation(obs, iterations=args.frontier_mask)
        cand = unk & near_obs                              # -> no outside-room / far-field fill
    pred_new = np.zeros(D, bool)
    nc = int(cand.sum())
    if n_gt and nc > 0:                                    # take the n_gt (or fewer) most-confident candidates
        k = min(n_gt, nc)
        uv = udf_pred[cand]; thr = np.partition(uv, k - 1)[k - 1]
        pred_new = cand & (udf_pred <= thr)
    if args.cc_filter and pred_new.any():                  # keep only prediction connected to observed structure
        lbl, _ = label(pred_new | obs)
        pred_new = pred_new & np.isin(lbl, np.unique(lbl[obs]))
    dist_true = distance_transform_edt(~true_surf)
    dist_pred = distance_transform_edt(~pred_new) if pred_new.any() else np.full(D, 1e6, np.float32)
    recall = 100.0 * float(np.mean(dist_pred[gt_new] <= 1.0)) if n_gt else 0.0
    dp = dist_true[pred_new]
    prec = 100.0 * float(np.mean(dp <= 1.0)) if pred_new.any() else 0.0
    halluc = 100.0 * float(np.mean(dp > 3.0)) if pred_new.any() else 0.0
    F = 2 * prec * recall / max(1e-9, prec + recall)
    print(f"[WHOLE-ROOM step{step}] F={F:.1f} recall={recall:.1f}% prec={prec:.1f}% "
          f"halluc={halluc:.1f}%  n_gt={n_gt}  windows={nwin}", flush=True)

    # --- coloured PLYs (blue=input, orange=model completion, green=GT) ---
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    def P(g, cap=400000):
        idx = np.argwhere(g).astype(np.float32)
        if len(idx) > cap:
            idx = idx[rng.choice(len(idx), cap, replace=False)]
        return idx * voxel
    name = Path(args.npz).stem
    np.savez_compressed(out / f"{name}_udfpred.npz", udf_pred=udf_pred.astype(np.float16))  # for offline re-scoring
    BLUE, ORANGE, GREEN = (60, 130, 210), (255, 122, 26), (30, 184, 79)
    write_ply(out / f"{name}_input.ply", [(P(obs), BLUE)])
    write_ply(out / f"{name}_pred_full.ply", [(P(obs), BLUE), (P(pred_new), ORANGE)])
    write_ply(out / f"{name}_gt_full.ply", [(P(obs), BLUE), (P(gt_new), GREEN)])
    print(f"wrote PLYs -> {out}", flush=True)


if __name__ == "__main__":
    main()
