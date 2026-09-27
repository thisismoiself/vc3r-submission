#!/usr/bin/env python3
"""Build DA3-depth geometry per office4 window (GT-free reference) and test it for:
  (1) outlier ranking  — does NOVA3R-pred->DA3 distance correlate with pred->GT error?
  (2) reference quality — how close is DA3's geometry to GT?
DA3 depth is unprojected with the GT poses so it lands in the same world frame as the
NOVA3R stitch prediction (per_window.npz). Run with --max-windows for a quick check.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image
from scipy.spatial import cKDTree
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r"); DA3SRC = REPO / "da3" / "src"
sys.path.insert(0, str(DA3SRC))
import types
for _m in ("pycolmap",):                       # colmap export unused; stub to avoid hard dep
    sys.modules.setdefault(_m, types.ModuleType(_m))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors

REPLICA = Path("/usr/prakt/s0016/Replica"); ROOM = "office4"
ap = argparse.ArgumentParser()
ap.add_argument("--stitch-dir", default="outputs/replica/stitch_office4_vel_lambda03_midpoint")
ap.add_argument("--max-windows", type=int, default=None)
ap.add_argument("--conf-pct", type=float, default=40.0)
args = ap.parse_args()

dev = "cuda"
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(dev).eval()

results = REPLICA / ROOM / "results"
poses = np.loadtxt(REPLICA / ROOM / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)   # c2w
cam = json.load(open(REPLICA / "cam_params.json"))["camera"]
K = np.array([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1]], np.float32)

sd = args.stitch_dir
z = np.load(f"{sd}/per_window.npz"); pred = z["pred"]
meta = json.load(open(f"{sd}/metrics.json"))["per_window"]
err = np.array([w["pred_to_gt"] for w in meta])
gt = np.asarray(trimesh.load(str(REPO/"outputs/replica/gt_pointclouds"/ROOM/f"{ROOM}_gt_2m.ply")).vertices)
gt_tree = cKDTree(gt)

W = len(meta) if args.max_windows is None else min(args.max_windows, len(meta))
rows = []
da3_clouds = []
for i in range(W):
    f0, f1 = meta[i]["f0"], meta[i]["f1"]
    fids = list(range(f0, f1 + 1, 10))                       # 8 frames (stride 10)
    paths = [str(results / f"frame{f:06d}.jpg") for f in fids]
    c2w = poses[fids]; w2c = np.linalg.inv(c2w)              # (N,4,4)
    Kin = np.tile(K[None], (len(fids), 1, 1))
    with torch.no_grad():
        p = model.inference(image=paths, extrinsics=w2c, intrinsics=Kin,
                            align_to_input_ext_scale=True, process_res=504)
    depth = np.asarray(p.depth); conf = None if p.conf is None else np.asarray(p.conf)
    Kp = np.asarray(p.intrinsics)                            # processed-res intrinsics
    Hh, Ww = depth.shape[-2:]
    imgs_u8 = np.zeros((len(fids), Hh, Ww, 3), np.uint8)     # colors unused
    # place DA3 depth in GT world frame via GT poses (w2c), processed intrinsics
    cthr = np.percentile(conf, args.conf_pct) if conf is not None else 0.0
    da3w, _ = _depths_to_world_points_with_colors(depth, Kp, w2c, imgs_u8, conf, cthr)
    da3w = da3w[np.isfinite(da3w).all(1)]
    da3_clouds.append(da3w)
    # metrics
    s = pred[i][np.random.default_rng(0).choice(len(pred[i]), 4000, False)]
    d_pred_da3 = cKDTree(da3w).query(s)[0].mean() if len(da3w) else np.nan        # GT-FREE signal
    d_da3_gt   = gt_tree.query(da3w[np.random.default_rng(0).choice(len(da3w), min(4000,len(da3w)), False)])[0].mean() if len(da3w) else np.nan
    rows.append((i, f0, f1, err[i], d_pred_da3, d_da3_gt, len(da3w)))
    print(f"win{i:2d} f{f0}-{f1}  pred->GT {err[i]:.3f}  pred->DA3 {d_pred_da3:.3f}  DA3->GT {d_da3_gt:.3f}  ({len(da3w):,} DA3 pts)", flush=True)

R = np.array([(r[3], r[4], r[5]) for r in rows])             # err, pred->DA3, DA3->GT
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std(); return (a*b).mean()
print(f"\nDA3->GT mean = {np.nanmean(R[:,2])*100:.1f}cm  (reference quality; lower=better)")
print(f"corr(pred->GT err, pred->DA3 dist) = {corr(R[:,0], R[:,1]):.3f}   (outlier-ranking power; >0.7 = usable)")
to = set(np.argsort(R[:,0])[::-1][:3]); fl = set(np.argsort(R[:,1])[::-1][:3])
print(f"top-3 by pred->DA3 = {sorted(fl)}  vs true outliers {sorted(to)}  hit={len(fl&to)}/3")
np.savez(f"{sd}/da3_ref.npz", **{f"da3_{i}": c for i, c in enumerate(da3_clouds)},
         err=R[:,0], pred_da3=R[:,1], da3_gt=R[:,2])
print(f"saved -> {sd}/da3_ref.npz")
