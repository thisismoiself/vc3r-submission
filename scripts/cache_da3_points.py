#!/usr/bin/env python3
"""Step-2 cache: DA3-predicted geometry per window, in the SAME normalized frame as
pts_norm / z_star, for the adapter's --geom-cond --geom-source da3.

For each window of an existing z_star cache we:
  1. read meta.pt (frame_ids, first_c2w, norm_factor=nf, flip),
  2. run DA3 on the window's ORIGINAL frames with GT poses -> predicted depth,
  3. unproject to the GT world frame (DA3 is GT-free; only poses/nf come from meta),
  4. map world -> first-camera frame (first_c2w), x-negate iff the window was flipped,
     divide by nf and *3  ==> exactly the pts_norm frame,
  5. confidence-filter + subsample, save da3_pts_norm.pt  (float16, [M,3]).

Frame contract mirrors cache_online_hungarian_zstar_windows.py:
    pts_norm = (world -> first_cam) [x-negate if flip] / nf * 3
so da3_pts_norm and pts_norm live in one frame and can be swapped as geom source.

We deliberately run DA3 on the *unflipped* images + GT poses (not X_FLIP): DA3's
align_to_input_ext_scale expects physical extrinsics, and the flip is a pure x-negation
we apply to the resulting points -- robust and scale-correct. Uses the stored GT nf so
training stays consistent with z_star; a novel-scene deployment would substitute a
DA3-derived nf (median norm of the DA3 first-cam cloud) instead.

NOTE: needs the GPU; do not run while a training job owns it. One-time pass.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, types
from PIL import Image

REPO = Path("/usr/prakt/s0016/vc3r")
sys.path.insert(0, str(REPO / "da3" / "src"))
sys.path.insert(0, str(REPO / "experiments" / "overfit_8frames"))
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
for _m in ("pycolmap",):                       # colmap export unused; stub to avoid hard dep
    sys.modules.setdefault(_m, types.ModuleType(_m))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
from multi_scene_train import world_to_first_camera


def fps_numpy(pts: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """Farthest-point subsample to m points (uniform spatial coverage of thin structures)."""
    n = len(pts)
    if n <= m:
        return pts
    rng = np.random.default_rng(seed)
    sel = np.empty(m, dtype=np.int64)
    sel[0] = rng.integers(n)
    d = np.full(n, np.inf)
    for i in range(1, m):
        d = np.minimum(d, ((pts - pts[sel[i - 1]]) ** 2).sum(1))
        sel[i] = int(d.argmax())
    return pts[sel]


def to_exact(pts: np.ndarray, m: int, use_fps: bool, fps_cap: int = 16384) -> np.ndarray:
    """Return exactly m points (subsample if more, resample-with-replacement pad if fewer),
    so every window's cloud stacks to a fixed shape at train time.

    For FPS, first random-subsample the candidates to fps_cap (like NOVA's _sample_features:
    random oversample -> FPS). Full FPS over ~1.3M unprojected points is ~140s/window in
    Python; capping candidates makes it ~2s while keeping near-uniform spatial coverage."""
    if len(pts) == 0:
        return np.zeros((m, 3), np.float32)
    if len(pts) >= m:
        if not use_fps:
            return pts[np.random.default_rng(0).choice(len(pts), m, replace=False)]
        cand = pts if len(pts) <= fps_cap else \
            pts[np.random.default_rng(0).choice(len(pts), fps_cap, replace=False)]
        return fps_numpy(cand, m)
    pad = np.random.default_rng(0).choice(len(pts), m - len(pts), replace=True)
    return np.concatenate([pts, pts[pad]], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True, type=Path,
                    help="Cache dir with <room>/<window>/meta.pt (e.g. scripts/data/fc_nf16_span24_100_l13).")
    ap.add_argument("--replica-root", type=Path, default=Path("/usr/prakt/s0016/Replica"))
    ap.add_argument("--rooms", nargs="*", default=None, help="Subset of room subdirs; default = all.")
    ap.add_argument("--out-points", type=int, default=4096, help="Points stored per window (train subsamples further).")
    ap.add_argument("--conf-pct", type=float, default=40.0, help="Drop DA3 points below this confidence percentile.")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--fps", action="store_true", help="Farthest-point subsample (default: uniform random).")
    ap.add_argument("--max-windows", type=int, default=None, help="Cap windows per room (smoke test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(dev).eval()
    cam = json.load(open(args.replica_root / "cam_params.json"))["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], np.float32)

    rooms = args.rooms or sorted(d.name for d in args.cache_root.iterdir() if d.is_dir())
    n_done = n_skip = 0
    for room in rooms:
        results = args.replica_root / room / "results"
        win_dirs = sorted(p for p in (args.cache_root / room).iterdir()
                          if p.is_dir() and (p / "meta.pt").exists())
        if args.max_windows is not None:
            win_dirs = win_dirs[:args.max_windows]
        for wd in win_dirs:
            out = wd / "da3_pts_norm.pt"
            if out.exists() and not args.overwrite:
                n_skip += 1; continue
            meta = torch.load(wd / "meta.pt", map_location="cpu", weights_only=False)
            fids = list(meta["frame_ids"])
            first_c2w = meta["first_c2w"].float()
            nf = float(meta["norm_factor"])
            flip = bool(meta.get("flip", False))

            # --- DA3 depth on the original frames, GT poses -> world points ---
            paths = [str(results / f"frame{f:06d}.jpg") for f in fids]
            c2w = np.stack([meta["poses_c2w"][i].numpy() for i in range(len(fids))]).astype(np.float32)
            w2c = np.linalg.inv(c2w)
            Kin = np.tile(K[None], (len(fids), 1, 1))
            with torch.no_grad():
                p = model.inference(image=paths, extrinsics=w2c, intrinsics=Kin,
                                    align_to_input_ext_scale=True, process_res=args.process_res)
            depth = np.asarray(p.depth); conf = None if p.conf is None else np.asarray(p.conf)
            Kp = np.asarray(p.intrinsics)
            H, W = depth.shape[-2:]
            cthr = np.percentile(conf, args.conf_pct) if conf is not None else 0.0
            da3_world, _ = _depths_to_world_points_with_colors(
                depth, Kp, w2c, np.zeros((len(fids), H, W, 3), np.uint8), conf, cthr)
            da3_world = da3_world[np.isfinite(da3_world).all(1)]

            # --- world -> pts_norm frame (mirror the cache exactly) ---
            first_cam = world_to_first_camera(torch.from_numpy(da3_world).float(), first_c2w).numpy()
            if flip:
                first_cam[:, 0] = -first_cam[:, 0]
            da3_norm = first_cam / nf * 3.0

            pts = to_exact(da3_norm, args.out_points, args.fps)   # exactly [M, 3]
            torch.save(torch.from_numpy(pts.astype(np.float16)).unsqueeze(0), out)  # [1, M, 3]

            # sanity: DA3 cloud should overlap the GT pts_norm in the same frame
            pn = torch.load(wd / "pts_norm.pt", map_location="cpu", weights_only=True)[0].numpy()
            from scipy.spatial import cKDTree
            d = cKDTree(pn).query(pts[np.random.default_rng(0).choice(len(pts), min(2000, len(pts)), False)])[0].mean()
            n_done += 1
            print(f"[{room}] {wd.name:<22} flip={int(flip)} nf={nf:.3f} "
                  f"da3={len(da3_norm):>7,}->{len(pts):>5,}  mean NN to pts_norm={d:.3f} (norm units)", flush=True)

    print(f"\ndone: wrote {n_done} windows, skipped {n_skip} existing. "
          f"(NN-to-pts_norm ~0.02-0.05 = frame OK; large => frame/flip bug)")


if __name__ == "__main__":
    main()
