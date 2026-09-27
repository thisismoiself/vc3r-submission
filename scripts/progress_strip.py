#!/usr/bin/env python3
"""Growing 'progress strip' for Stage-A: sample the SAME held-out val room + SAME crop at each
checkpoint milestone and lay the model completions side by side, so you can watch the occluded
surface sharpen (or not) over training. Bookended by the fixed INPUT (partial scan) and GT panels.

Reproduces train_diffusion's exact val split (sorted glob -> shuffle seed 0 -> [:max_scenes] ->
first val_frac = val), locks room+crop, and renders each MODEL panel at an identical viewpoint.

Frames are cached to <out>/frames/step_<N>.npz so a job restart never loses history; the strip PNG
is rebuilt from all frames every time a new one lands.

  watch mode:  poll the live checkpoint; whenever step crosses a new multiple of --milestone that
               isn't rendered yet, sample + append a frame + rebuild the strip.
"""
import argparse, glob, sys, time, json, shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import TSDFCropDataset, VOXEL, BAND, UNKNOWN, OBSERVED
from train_diffusion import unpack, DEV
from diffcomplete_ddpm import DiffCompleteUNet, Diffusion


def val_rooms(cache_dir, max_scenes=1600, val_frac=0.03):
    caches = sorted(glob.glob(f"{cache_dir}/*.npz"))
    rng = np.random.default_rng(0); rng.shuffle(caches)      # mirror train_diffusion exactly
    caches = caches[:max_scenes]
    n_val = max(1, int(len(caches) * val_frac))
    return caches[:n_val]


def fixed_crop(room_npz, tries=8, seed=0):
    """Deterministic frontier crop with the most unknown near-surface target voxels."""
    ds = TSDFCropDataset([room_npz], band=BAND, samples_per_epoch=tries, seed=seed)
    best, best_n = None, -1
    for i in range(tries):
        inp, gt, msk = ds[i]
        x0, cond, mask, near = unpack(tuple(t[None] for t in (inp, gt, msk)))
        n = int(((mask == UNKNOWN) & near).sum())
        if n > best_n:
            best_n, best = n, (x0, cond, mask, near)
    return best


def surf_pts(bool_grid, voxel, rng, cap=40000):
    idx = np.argwhere(bool_grid)
    if len(idx) > cap:
        idx = idx[rng.choice(len(idx), cap, replace=False)]
    return idx * voxel


def base_geometry(x0, cond, mask, near):
    """Checkpoint-independent point sets: observed input surface + GT occluded surface. Returns
    dict with obs pts, gt-new pts, and everything needed to score/render a model prediction."""
    surf = 0.7 * VOXEL
    m = mask[0, 0].cpu().numpy()
    part = np.abs((cond[:, 0].float() * BAND)[0].cpu().numpy())
    udf_gt = ((x0.float() + 1) * 0.5 * BAND)[0, 0].cpu().numpy()
    obs = (m == OBSERVED) & (part < surf)
    gt_new = (m == UNKNOWN) & (udf_gt < surf)
    true_surf = obs | gt_new                                    # every REAL surface voxel in the crop
    # precompute (once) distance in voxels from every voxel to the nearest real surface -> lets each
    # frame measure PRECISION (are predicted points near a real surface?) and the hallucination rate.
    dist_true = distance_transform_edt(~true_surf)
    return dict(obs=obs, gt_new=gt_new, unk=(m == UNKNOWN), udf_gt=udf_gt,
                near=near[0, 0].cpu().numpy(), n_gt=int(gt_new.sum()), dist_true=dist_true)


def sample_frame(model, diff, x0, cond, mask, near, base, steps=50):
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad(), torch.autocast("cuda", dtype=amp):
        pred = diff.ddim_sample(model, cond, mask=mask, x_known=(2 * cond[:, :1].abs() - 1),
                                steps=steps, replace=True)
    udf_pred = ((pred.float() + 1) * 0.5 * BAND).clamp(0, BAND)[0, 0].cpu().numpy()
    unk, n_gt = base["unk"], base["n_gt"]
    # model surface = the n_gt most-confident unknown voxels (rank by predicted distance) -> matches
    # GT density, so the picture reflects true recall/precision not a threshold artifact.
    pred_new = np.zeros_like(unk)
    if n_gt and unk.sum() > n_gt:
        uv = udf_pred[unk]; thr = np.partition(uv, n_gt - 1)[n_gt - 1]
        pred_new = unk & (udf_pred <= thr)
    m = unk & base["near"]
    mae = float(np.abs(udf_pred - base["udf_gt"])[m].mean() * 100) if m.any() else float("nan")
    # --- two-sided metrics with a 1-voxel tolerance (honest: precision penalises off-surface points) ---
    gt_new = base["gt_new"]; dist_true = base["dist_true"]
    dist_pred = distance_transform_edt(~pred_new) if pred_new.any() else np.full_like(dist_true, 1e6)
    recall = 100.0 * float(np.mean(dist_pred[gt_new] <= 1.0)) if n_gt else 0.0   # GT covered by a pred
    dp = dist_true[pred_new]                                                      # pred dist to any real surf
    prec = 100.0 * float(np.mean(dp <= 1.0)) if pred_new.any() else 0.0           # pred near a real surface
    halluc = 100.0 * float(np.mean(dp > 3.0)) if pred_new.any() else 0.0          # pred >6cm from any surface
    fscore = 2 * prec * recall / max(1e-9, prec + recall)
    return pred_new, dict(mae=mae, recall=recall, prec=prec, F=fscore, halluc=halluc), udf_pred


def _scatter(ax, groups, elev, azim, title):
    for pts, c, s, a in groups:
        if len(pts):
            ax.scatter(pts[:, 0], -pts[:, 2], pts[:, 1], c=c, s=s, alpha=a, linewidths=0, marker=".")  # -Z keeps right-handed (no mirror), Y up
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.6)); ax.axis("off")


def rebuild_strip(frames_dir, base_npz, out_png, elev=20, azim=-70):
    rng = np.random.default_rng(0)
    b = np.load(base_npz)
    voxel = float(b["voxel"])
    obs = surf_pts(b["obs"], voxel, rng); gt = surf_pts(b["gt_new"], voxel, rng)
    frames = sorted(glob.glob(f"{frames_dir}/step_*.npz"), key=lambda p: int(Path(p).stem.split("_")[1]))
    n = len(frames)
    fig = plt.figure(figsize=(4 * (n + 2), 4.5))
    ax = fig.add_subplot(1, n + 2, 1, projection="3d")
    _scatter(ax, [(obs, "#2f7fd0", 3, .5)], elev, azim, "INPUT\npartial scan")
    for j, f in enumerate(frames):
        d = np.load(f); pn = surf_pts(d["pred_new"], voxel, rng)
        ax = fig.add_subplot(1, n + 2, j + 2, projection="3d")
        F = f"F {float(d['F']):.0f}  " if "F" in d else ""       # older frames lack the new metrics
        prec = f"  P {float(d['prec']):.0f}%" if "prec" in d else ""
        _scatter(ax, [(obs, "#2f7fd0", 3, .35), (pn, "#ff7a1a", 5, .85)], elev, azim,
                 f"step {int(d['step'])//1000}k\n{F}R {float(d['recall']):.0f}%{prec}")
    ax = fig.add_subplot(1, n + 2, n + 2, projection="3d")
    _scatter(ax, [(obs, "#2f7fd0", 3, .35), (gt, "#1eb84f", 5, .85)], elev, azim, "GROUND TRUTH\ncomplete")
    fig.suptitle(f"Stage-A held-out completion over training  ({voxel*100:.0f}cm grid, model-filled = orange)",
                 fontsize=12, y=0.02)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close()
    print(f"[strip] {n} frames -> {out_png}", flush=True)


def load_ema(ckpt):
    d = torch.load(ckpt, map_location=DEV, weights_only=False)
    model = DiffCompleteUNet(use_ckpt=False).to(DEV)
    model.load_state_dict(d["ema"]); model.eval()
    return model, int(d.get("step", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/diffusion_stageA.pt")
    ap.add_argument("--cache-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/experiments/overfit_8frames/progress_strip")
    ap.add_argument("--extra-ckpts", nargs="*", default=[], help="static snapshot ckpts to seed early frames")
    ap.add_argument("--milestone", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--room-idx", type=int, default=0, help="which val room to lock onto")
    ap.add_argument("--poll-min", type=float, default=15.0)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--history-dir", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/snapshots/history")
    ap.add_argument("--history-keep", type=int, default=8, help="rolling # of raw checkpoints to retain")
    ap.add_argument("--track-rooms", type=int, nargs="*", default=[1, 3, 5], help="hard rooms for best-tracking")
    ap.add_argument("--val-frac", type=float, default=0.03, help="frac of cache-dir treated as val (use 1.0 for a dedicated holdout dir)")
    ap.add_argument("--band", type=float, default=0.0, help="override truncation band (m) to match the model/cache, e.g. 0.06; 0=default")
    ap.add_argument("--best-ckpt", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/snapshots/stageA_BEST.pt")
    ap.add_argument("--best-json", default="/usr/prakt/s0016/vc3r/outputs/consecutive_windows/snapshots/stageA_BEST.json")
    args = ap.parse_args()

    if args.band > 0:                                           # match the model/cache band (v2 = 0.06)
        global BAND
        BAND = args.band
        import train_diffusion as _td                           # unpack() lives there, uses its globals
        _td.BAND = args.band; _td.SURF_BAND_M = args.band - VOXEL
        print(f"[band] eval band -> {BAND} m", flush=True)

    out = Path(args.out_dir); frames_dir = out / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    base_npz = out / "base.npz"; strip_png = out / "progress_strip.png"

    vr = val_rooms(args.cache_dir, val_frac=args.val_frac)
    room = vr[min(args.room_idx, len(vr) - 1)]
    print(f"[lock] held-out room = {Path(room).stem[:40]}  ({len(vr)} val rooms)", flush=True)
    x0, cond, mask, near = fixed_crop(room, seed=0)
    base = base_geometry(x0, cond, mask, near)
    np.savez_compressed(base_npz, obs=base["obs"], gt_new=base["gt_new"], voxel=VOXEL)
    print(f"[lock] crop locked: {base['n_gt']} GT occluded-surface voxels", flush=True)

    diff = Diffusion(T=1000, device=DEV)

    # lock a few HARD held-out rooms too, for cross-room best-checkpoint tracking (the run oscillates,
    # so we must auto-capture the peak checkpoints by a precision-aware metric, not lucky manual evals).
    track = {}
    for r in args.track_rooms:
        try:
            tx0, tcond, tmask, tnear = fixed_crop(vr[min(r, len(vr) - 1)], seed=0)
            track[r] = (tx0, tcond, tmask, tnear, base_geometry(tx0, tcond, tmask, tnear))
        except Exception as e:
            print(f"[track] room {r} load failed ({e})", flush=True)
    best = {"F": -1.0, "step": -1}
    if Path(args.best_json).exists():
        try: best = json.load(open(args.best_json))
        except Exception: pass
    print(f"[track] {len(track)} tracked rooms {list(track)}; best-so-far F={best['F']:.1f}@{best['step']}",
          flush=True)

    def render_ckpt(ckpt):
        model, step = load_ema(ckpt)
        ms = (step // args.milestone) * args.milestone           # snap to milestone label
        fpath = frames_dir / f"step_{ms:06d}.npz"
        if fpath.exists():
            print(f"[skip] step {ms} already rendered", flush=True); return step
        pred_new, mt, _ = sample_frame(model, diff, x0, cond, mask, near, base, args.steps)
        np.savez_compressed(fpath, pred_new=pred_new, step=ms, **mt)
        print(f"[frame] step {ms}: F={mt['F']:.1f} recall={mt['recall']:.1f}% prec={mt['prec']:.1f}% "
              f"halluc={mt['halluc']:.1f}% MAE={mt['mae']:.2f}cm -> {fpath.name}", flush=True)
        # cross-room best-tracking: eval the hard rooms, save the checkpoint if it's the best yet
        if track:
            try:
                Fs = []
                for r, (tx0, tc, tm, tn, tb) in track.items():
                    _, tmt, _ = sample_frame(model, diff, tx0, tc, tm, tn, tb, args.steps)
                    Fs.append(tmt["F"])
                meanF = float(np.mean(Fs))
                print(f"[track] step {ms}: cross-room F={meanF:.1f}  ({'/'.join(f'{x:.0f}' for x in Fs)})",
                      flush=True)
                if meanF > best["F"]:
                    best.update(F=meanF, step=ms)
                    shutil.copy2(ckpt, args.best_ckpt)
                    json.dump(best, open(args.best_json, "w"))
                    print(f"[best] NEW BEST cross-room F={meanF:.1f} @ step{ms} -> {Path(args.best_ckpt).name}",
                          flush=True)
            except Exception as e:
                print(f"[track] eval failed ({e})", flush=True)
        del model; torch.cuda.empty_cache()
        return step

    for c in args.extra_ckpts:                                   # seed early frames from snapshots
        if Path(c).exists():
            render_ckpt(c)
    render_ckpt(args.ckpt)
    rebuild_strip(frames_dir, base_npz, strip_png)

    if not args.watch:
        return
    last_ms = max(int(Path(f).stem.split("_")[1]) for f in glob.glob(f"{frames_dir}/step_*.npz"))
    print(f"[watch] polling every {args.poll_min} min for next {args.milestone}-step milestone "
          f"(last={last_ms})", flush=True)
    while True:
        time.sleep(args.poll_min * 60)
        try:
            step = int(torch.load(args.ckpt, map_location="cpu", weights_only=False).get("step", 0))
        except Exception as e:
            print(f"[watch] ckpt read failed ({e}); retry", flush=True); continue
        if step - last_ms >= args.milestone:
            render_ckpt(args.ckpt)
            rebuild_strip(frames_dir, base_npz, strip_png)
            last_ms = (step // args.milestone) * args.milestone
            # preserve a rolling history of raw checkpoints so a peak is never lost again (the run
            # had no best-checkpoint tracking and overwrote its ~70k peak). Keep the last N.
            if args.history_dir:
                hd = Path(args.history_dir); hd.mkdir(parents=True, exist_ok=True)
                dst = hd / f"stageA_step{last_ms//1000}k.pt"
                if not dst.exists():
                    shutil.copy2(args.ckpt, dst)                  # shutil imported at module top
                    print(f"[history] preserved {dst.name}", flush=True)
                    hist = sorted(hd.glob("stageA_step*k.pt"), key=lambda p: int(p.stem.split("step")[1][:-1]))
                    for old in hist[:-args.history_keep]:
                        old.unlink(); print(f"[history] rotated out {old.name}", flush=True)


if __name__ == "__main__":
    main()
