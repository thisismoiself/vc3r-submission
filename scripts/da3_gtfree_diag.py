#!/usr/bin/env python3
"""How good are DA3's GT-FREE predicted poses + metric scale on office4 windows?

Per window, run DA3 twice:
  (GT)      extrinsics = GT w2c  (current pipeline)
  (GT-free) extrinsics = None    (DA3 predicts poses + intrinsics + metric depth)
Compare, in a global-frame-invariant way:
  - relative-rotation error  : geodesic angle between DA3-pred and GT frame-to-frame rotations
  - metric-scale ratio       : DA3-pred camera baseline / GT camera baseline (1.0 = perfect metric)
  - reference geometry        : DA3-free cloud -> GT distance after Umeyama(sim) aligning the
                                DA3-pred camera centers to the GT centers (isolates geometry, not frame)
Tells us whether a fully GT-free stitch (cam_token + scale + placement from DA3) is viable.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, trimesh
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r"); sys.path.insert(0, str(REPO / "da3" / "src"))
import types
for m in ("pycolmap",): sys.modules.setdefault(m, types.ModuleType(m))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
from scipy.spatial import cKDTree

REPLICA = Path("/usr/prakt/s0016/Replica"); ROOM = "office4"


def rot_geo_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = (np.trace(R) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def umeyama_sim(src, dst):
    # similarity transform (scale s, R, t) mapping src->dst (both (N,3))
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    var = (S ** 2).sum() / len(src)
    s = np.trace(np.diag(d)) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-dir", default="outputs/replica/stitch_office4_complete_office4")
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    dev = "cuda"
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(dev).eval()
    results = REPLICA / ROOM / "results"
    poses = np.loadtxt(REPLICA / ROOM / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)  # c2w GT
    cam = json.load(open(REPLICA / "cam_params.json"))["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], np.float32)
    gt = np.asarray(trimesh.load(str(REPO / "outputs/replica/gt_pointclouds" / ROOM / f"{ROOM}_gt_2m.ply")).vertices)
    gt_tree = cKDTree(gt)

    meta = json.load(open(f"{args.stitch_dir}/metrics.json"))["per_window"]
    W = min(args.max_windows, len(meta))
    rot_errs, scale_ratios, geo_free, geo_gt = [], [], [], []

    for i in range(W):
        f0, f1 = meta[i]["f0"], meta[i]["f1"]
        fids = list(range(f0, f1 + 1, args.stride))
        paths = [str(results / f"frame{f:06d}.jpg") for f in fids]
        c2w_gt = poses[fids]; w2c_gt = np.linalg.inv(c2w_gt)
        Kin = np.tile(K[None], (len(fids), 1, 1))
        with torch.no_grad():
            pf = model.inference(image=paths, extrinsics=None, intrinsics=None, process_res=392)   # GT-FREE
            pg = model.inference(image=paths, extrinsics=w2c_gt, intrinsics=Kin,
                                 align_to_input_ext_scale=True, process_res=392)                    # GT ref
        # DA3-pred extrinsics (N,3,4) w2c -> pad to (N,4,4) -> c2w
        ex_free = np.asarray(pf.extrinsics)                       # w2c (N,3,4)
        if ex_free.shape[-2:] == (3, 4):
            bottom = np.tile(np.array([0, 0, 0, 1], np.float32), (len(ex_free), 1, 1))
            ex_free = np.concatenate([ex_free, bottom], axis=1)   # (N,4,4)
        c2w_free = np.linalg.inv(ex_free)
        cen_free = c2w_free[:, :3, 3]; cen_gt = c2w_gt[:, :3, 3]
        # relative-rotation error (frame 0 -> k), global-frame invariant
        re = [rot_geo_deg(c2w_gt[0, :3, :3].T @ c2w_gt[k, :3, :3],
                          c2w_free[0, :3, :3].T @ c2w_free[k, :3, :3]) for k in range(1, len(fids))]
        rot_errs.append(np.mean(re))
        # metric scale: mean consecutive-baseline ratio
        bl_gt = np.linalg.norm(np.diff(cen_gt, axis=0), axis=1).mean()
        bl_free = np.linalg.norm(np.diff(cen_free, axis=0), axis=1).mean()
        scale_ratios.append(bl_free / (bl_gt + 1e-9))
        # DA3-free cloud -> GT after Umeyama(cameras) alignment
        s, R, t = umeyama_sim(cen_free, cen_gt)
        depth_f = np.asarray(pf.depth); Kp_f = np.asarray(pf.intrinsics)
        u8 = np.zeros((len(fids), *depth_f.shape[-2:], 3), np.uint8)
        cf, _ = _depths_to_world_points_with_colors(depth_f, Kp_f, ex_free, u8, None, 0.0)
        cf = cf[np.isfinite(cf).all(1)]
        cf_al = (s * (R @ cf.T).T + t)
        idx = np.random.default_rng(0).choice(len(cf_al), min(20000, len(cf_al)), False)
        geo_free.append(gt_tree.query(cf_al[idx])[0].mean())
        # GT-posed DA3 cloud -> GT (reference quality with GT poses)
        depth_g = np.asarray(pg.depth); Kp_g = np.asarray(pg.intrinsics)
        cg, _ = _depths_to_world_points_with_colors(depth_g, Kp_g, w2c_gt, u8, None, 0.0)
        cg = cg[np.isfinite(cg).all(1)]
        idx2 = np.random.default_rng(0).choice(len(cg), min(20000, len(cg)), False)
        geo_gt.append(gt_tree.query(cg[idx2])[0].mean())
        print(f"win{i:2d} f{f0}-{f1}  relRot={rot_errs[-1]:5.2f}deg  scale(pred/GT)={scale_ratios[-1]:.3f}  "
              f"DA3free->GT={geo_free[-1]*100:5.1f}cm  DA3gt->GT={geo_gt[-1]*100:4.1f}cm", flush=True)

    print("\n===== GT-FREE DA3 feasibility (mean over %d windows) =====" % W)
    print(f"  relative-rotation error : {np.mean(rot_errs):.2f} +- {np.std(rot_errs):.2f} deg")
    print(f"  metric-scale ratio      : {np.mean(scale_ratios):.3f} +- {np.std(scale_ratios):.3f}  (1.0=perfect)")
    print(f"  DA3-free geom -> GT     : {np.mean(geo_free)*100:.1f} +- {np.std(geo_free)*100:.1f} cm  (poses predicted)")
    print(f"  DA3-GT   geom -> GT     : {np.mean(geo_gt)*100:.1f} +- {np.std(geo_gt)*100:.1f} cm  (poses given)")


if __name__ == "__main__":
    main()
