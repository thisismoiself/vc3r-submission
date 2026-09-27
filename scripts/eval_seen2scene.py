#!/usr/bin/env python3
"""Score a Seen2Scene large-scale-completion output against the SAME clean-mesh GT + DA3 reference as
scripts/eval_replica_v3.py, so Seen2Scene is directly comparable to our v3 / v5 whole-room numbers.

Seen2Scene emits a full completed surface mesh ("mesh_tsdf (Generation)_0.ply"); we sample it to points
and compute Chamfer / F@1 / F@5 / acc / comp vs GT, both RAW and min-area-OBB clipped (v3/v5 report the
clipped number). Deep/shallow hole recall uses the same DA3-hole definition. GT points, DA3 points, voxel,
and the OBB reference all come from our 2cm cache — identical to eval_replica_v3.
"""
import sys, argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from eval_replica_v3 import pts_from, obb_clip, chamfer_cm, fscore, iou_vox   # identical metric code
OBS, FREE, UNK = 0, 1, 2


def sample_mesh(path, n=800000):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    if len(m.faces) == 0:
        return np.asarray(m.vertices)
    pts, _ = trimesh.sample.sample_surface(m, n)
    return np.asarray(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--gen-mesh", required=True)
    ap.add_argument("--cache-dir", default="outputs/tsdf_cache_replica_da3_006")
    args = ap.parse_args()

    c = np.load(f"{args.cache_dir}/replica_{args.room}.npz")
    mask = c["mask"]; gt = np.abs(c["gt_tsdf"].astype(np.float32)); pt = np.abs(c["partial_tsdf"].astype(np.float32))
    origin = c["origin"].astype(np.float64); voxel = float(c["voxel"]); surf = 0.7 * voxel

    gt_pts = pts_from(gt < surf, origin, voxel)
    da3_pts = pts_from((mask == OBS) & (pt < surf), origin, voxel)
    pred = sample_mesh(args.gen_mesh)

    # alignment sanity: pred bbox vs gt bbox (Seen2Scene should output in world coords)
    print(f"[{args.room}] pred bbox {pred.min(0).round(2)}..{pred.max(0).round(2)} | "
          f"gt bbox {gt_pts.min(0).round(2)}..{gt_pts.max(0).round(2)}  (pred n={len(pred)})")

    # GT-free deployment clip: min-area OBB in XY (as in eval_replica_v3) PLUS a vertical clip
    # to the observed (DA3) Z extent + 5cm margin. The taller completion canvas can hallucinate
    # geometry above the ceiling / below the floor; clipping to the observed room height removes it,
    # analogous to the XY OBB clip (both use only the DA3-observed points, so still GT-free).
    zmar = 0.05
    zlo, zhi = da3_pts[:, 2].min() - zmar, da3_pts[:, 2].max() + zmar
    pred_zc = pred[(pred[:, 2] >= zlo) & (pred[:, 2] <= zhi)]
    pred_clip = obb_clip(pred_zc, da3_pts)

    # deep/shallow hole recall vs the DA3 partial (same definition as eval_replica_v3)
    d_gt_da3 = cKDTree(da3_pts).query(gt_pts, workers=4)[0]
    shallow = gt_pts[(d_gt_da3 > 3 * voxel) & (d_gt_da3 <= 25 * voxel)]
    deep = gt_pts[d_gt_da3 > 25 * voxel]
    rec = lambda H, P: 100 * np.mean(cKDTree(P).query(H, workers=4)[0] <= 2 * voxel) if len(H) else 0.0

    print(f"{'variant':16s} {'Chamfer':>8} {'F@1':>6} {'F@5':>6} {'acc@5':>6} {'comp@5':>6}")
    for name, P in [("Seen2Scene-raw", pred), ("Seen2Scene-OBB", pred_clip)]:
        f1, p1, r1 = fscore(P, gt_pts, 0.01)
        f5, p5, r5 = fscore(P, gt_pts, 0.05)
        ch = chamfer_cm(P, gt_pts)
        print(f"{name:16s} {ch:7.2f}cm {f1:5.1f} {f5:5.1f} {p5:5.1f}% {r5:5.1f}%")
    print(f"deep-hole recall {rec(deep, pred_clip):.0f}%  shallow {rec(shallow, pred_clip):.0f}%  "
          f"(n_deep={len(deep)}, n_shallow={len(shallow)})")


if __name__ == "__main__":
    main()
