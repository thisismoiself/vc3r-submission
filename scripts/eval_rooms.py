#!/usr/bin/env python3
"""Evaluate a Stage-A checkpoint on SEVERAL held-out val rooms (not just the strip room) at the
current step. For each room: lock a frontier crop, run the DDPM/DDIM sampler, report the honest
two-sided metrics (F / recall / precision / hallucination), and dump a completion npz + coloured
PLYs. Also renders a montage (rows = rooms, cols = INPUT | MODEL | GT).

Reuses the exact metric definitions from progress_strip so numbers are directly comparable."""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import VOXEL
from train_diffusion import DEV
from diffcomplete_ddpm import Diffusion
from progress_strip import (val_rooms, fixed_crop, base_geometry, sample_frame, load_ema,
                            surf_pts, _scatter)
from export_ply import write_ply
from tsdf_dataset import BAND, UNKNOWN


def dense_pts(bool_grid, voxel, rng, jitter=6):
    """Voxel-center points BLOWN UP for visualisation: emit `jitter` points uniformly inside each
    surface voxel's cube. Purely cosmetic (fills the 2cm lattice into a solid-looking surface) —
    metrics are computed separately on the honest voxel grid, never on these."""
    idx = np.argwhere(bool_grid).astype(np.float32)
    if not len(idx):
        return idx.reshape(0, 3)
    reps = np.repeat(idx, jitter, axis=0)
    reps += (rng.random(reps.shape) - 0.5)                       # +/- 0.5 voxel within the cell
    return reps * voxel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/diffusion_stageA.pt")
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/experiments/overfit_8frames/eval_rooms")
    ap.add_argument("--room-idxs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--hero-room", type=int, default=-1, help="room idx for the dense export; -1 = best F")
    ap.add_argument("--jitter", type=int, default=6, help="points emitted per surface voxel in dense PLY")
    args = ap.parse_args()

    out = Path(args.out_dir); (out / "ply").mkdir(parents=True, exist_ok=True)
    vr = val_rooms(args.cache_dir)
    model, step = load_ema(args.ckpt)
    diff = Diffusion(T=1000, device=DEV)
    print(f"[ckpt] step {step}, EMA loaded, evaluating {len(args.room_idxs)} held-out rooms\n", flush=True)
    BLUE, ORANGE, GREEN = (60, 130, 210), (255, 130, 30), (30, 200, 90)

    results, panels, keep = [], [], {}
    for ridx in args.room_idxs:
        room = vr[min(ridx, len(vr) - 1)]; name = Path(room).stem[:36]
        x0, cond, mask, near = fixed_crop(room, seed=0)
        base = base_geometry(x0, cond, mask, near)
        pred_new, mt, udf_pred = sample_frame(model, diff, x0, cond, mask, near, base, args.steps)
        results.append((ridx, name, base["n_gt"], mt))
        print(f"[room {ridx}] {name}\n   n_gt={base['n_gt']:>6}  F={mt['F']:.1f}  recall={mt['recall']:.1f}%  "
              f"prec={mt['prec']:.1f}%  halluc={mt['halluc']:.1f}%  MAE={mt['mae']:.2f}cm", flush=True)
        # PLYs (voxel-centre, honest matched density)
        obs_p = np.argwhere(base["obs"]).astype(np.float32) * VOXEL
        mdl_p = np.argwhere(pred_new).astype(np.float32) * VOXEL
        gt_p = np.argwhere(base["gt_new"]).astype(np.float32) * VOXEL
        write_ply(out / "ply" / f"room{ridx}_completion.ply", [(obs_p, BLUE), (mdl_p, ORANGE)])
        write_ply(out / "ply" / f"room{ridx}_overlay.ply", [(obs_p, BLUE), (gt_p, GREEN), (mdl_p, ORANGE)])
        panels.append((name, base, pred_new, mt))
        keep[ridx] = (name, base, udf_pred, mt)

    # ---- DENSE hero export for the best-F room (or user-chosen) ----
    hero = args.hero_room if args.hero_room in keep else max(keep, key=lambda k: keep[k][3]["F"])
    name, base, udf_pred, mt = keep[hero]
    rng = np.random.default_rng(0)
    # full predicted occluded surface (not density-capped) = every unknown voxel within ~1 voxel of surface
    model_surf = base["unk"] & (udf_pred < 0.9 * VOXEL)
    obs_d = dense_pts(base["obs"], VOXEL, rng, args.jitter)
    mdl_d = dense_pts(model_surf, VOXEL, rng, args.jitter)
    gt_d = dense_pts(base["gt_new"], VOXEL, rng, args.jitter)
    write_ply(out / "ply" / f"HERO_room{hero}_completion_dense.ply", [(obs_d, BLUE), (mdl_d, ORANGE)])
    write_ply(out / "ply" / f"HERO_room{hero}_overlay_dense.ply", [(obs_d, BLUE), (gt_d, GREEN), (mdl_d, ORANGE)])
    print(f"[hero] room {hero} ({name})  F={mt['F']:.1f} prec={mt['prec']:.1f}%  "
          f"dense: obs={len(obs_d)} model={len(mdl_d)} gt={len(gt_d)} pts (jitter={args.jitter})", flush=True)

    # montage: one row per room, columns INPUT | MODEL | GT
    rng = np.random.default_rng(0); nR = len(panels)
    fig = plt.figure(figsize=(12, 4 * nR))
    for r, (name, base, pred_new, mt) in enumerate(panels):
        o = surf_pts(base["obs"], VOXEL, rng); pn = surf_pts(pred_new, VOXEL, rng)
        gn = surf_pts(base["gt_new"], VOXEL, rng)
        for c, (grp, ttl) in enumerate([([(o, "#2f7fd0", 3, .5)], f"{name[:22]}\nINPUT"),
                                        ([(o, "#2f7fd0", 3, .35), (pn, "#ff7a1a", 5, .85)],
                                         f"MODEL  F={mt['F']:.0f} R={mt['recall']:.0f}% P={mt['prec']:.0f}%"),
                                        ([(o, "#2f7fd0", 3, .35), (gn, "#1eb84f", 5, .85)], "GROUND TRUTH")]):
            ax = fig.add_subplot(nR, 3, r * 3 + c + 1, projection="3d")
            _scatter(ax, grp, 20, -70, ttl)
    fig.suptitle(f"Stage-A step {step//1000}k — {nR} unseen val rooms (model-filled = orange)", y=0.005, fontsize=13)
    plt.tight_layout(rect=(0, 0.02, 1, 0.99))
    plt.savefig(out / f"eval_rooms_step{step//1000}k.png", dpi=110, bbox_inches="tight"); plt.close()

    # summary
    Fs = [r[3]["F"] for r in results]; Ps = [r[3]["prec"] for r in results]; Rs = [r[3]["recall"] for r in results]
    print(f"\n[summary] step {step//1000}k over {len(results)} rooms: "
          f"F={np.mean(Fs):.1f}  recall={np.mean(Rs):.1f}%  prec={np.mean(Ps):.1f}%", flush=True)
    print(f"-> montage: {out}/eval_rooms_step{step//1000}k.png", flush=True)


if __name__ == "__main__":
    main()
