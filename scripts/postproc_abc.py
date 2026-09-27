#!/usr/bin/env python3
"""A+B+C post-processing for whole-room DiffComplete predictions, plus a plain bbox clip, scored
against the old frontier-mask. No GPU / no re-sampling.

  A  alpha-shape room clip : keep prediction inside the observed floor-plan concave hull (2D alpha
                            shape of observed XY, extruded over observed Z). Kills EXTERIOR spray,
                            robust to L-shaped rooms (a bbox would leak into the concave notch).
  B  free-space veto       : drop predicted voxels sitting in confidently-seen-empty space (a large
                            fraction of their local neighborhood is carved FREE) = interior floaters.
  C  plane-preserve        : RANSAC dominant planes (floor/ceiling/walls) from observed; predictions
                            ON those planes are ALWAYS kept (even deep), so big planar hole-fills
                            survive at high precision.
final = (inside_room  &  ~free_floater)  |  (on_plane & inside_room)
"""
import sys, argparse
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import uniform_filter
OBS, FREE, UNK = 0, 1, 2


def alpha_shape_2d(pts_xy, alpha):
    """Boolean-tester for the 2D concave hull: Delaunay minus triangles with any edge > alpha."""
    if len(pts_xy) > 40000:
        pts_xy = pts_xy[np.random.default_rng(0).choice(len(pts_xy), 40000, replace=False)]
    tri = Delaunay(pts_xy)
    s = tri.simplices
    e = np.stack([np.linalg.norm(pts_xy[s[:, 0]] - pts_xy[s[:, 1]], axis=1),
                  np.linalg.norm(pts_xy[s[:, 1]] - pts_xy[s[:, 2]], axis=1),
                  np.linalg.norm(pts_xy[s[:, 2]] - pts_xy[s[:, 0]], axis=1)], 1)
    valid = e.max(1) <= alpha
    return tri, valid


def ransac_planes(pts, n_planes=6, thr=0.03, iters=400, min_inl=4000):
    """Greedy RANSAC: peel off up to n_planes dominant planes. Returns list of (normal, d)."""
    rng = np.random.default_rng(0)
    remain = pts.copy(); planes = []
    for _ in range(n_planes):
        if len(remain) < min_inl:
            break
        best_inl, best_pl = 0, None
        for _ in range(iters):
            i = rng.choice(len(remain), 3, replace=False)
            p = remain[i]; n = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n = n / nn; d = -n @ p[0]
            inl = np.abs(remain @ n + d) < thr
            c = int(inl.sum())
            if c > best_inl:
                best_inl, best_pl, best_mask = c, (n, d), inl
        if best_pl is None or best_inl < min_inl:
            break
        planes.append(best_pl); remain = remain[~best_mask]
    return planes


def surf_pts(mask, tsdf, which, surf, origin, voxel):
    sel = (mask == which) & (np.abs(tsdf) < surf)
    idx = np.argwhere(sel)
    return origin + idx * voxel, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--udfpred", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="experiments/overfit_8frames/coarse_pcd")
    ap.add_argument("--surf-thr", type=float, default=0.03, help="udf<thr (m) = predicted surface")
    args = ap.parse_args()

    u = np.load(args.udfpred)["udf_pred"].astype(np.float32)
    c = np.load(args.cache)
    mask = c["mask"]; gt = c["gt_tsdf"].astype(np.float32); pt = c["partial_tsdf"].astype(np.float32)
    origin = c["origin"].astype(np.float64); voxel = float(c["voxel"]); band = float(c["band"])
    surf = 0.7 * voxel

    # ---- geometry ----
    obs_pts, obs_sel = surf_pts(mask, pt, OBS, surf, origin, voxel)
    gt_all_pts, gt_sel = surf_pts(mask, gt, None if False else 999, surf, origin, voxel)  # placeholder
    gt_sel = (np.abs(gt) < surf); gt_pts = origin + np.argwhere(gt_sel) * voxel
    # base prediction: unknown voxels with small predicted udf
    base = (mask == UNK) & (u < args.surf_thr)
    base_idx = np.argwhere(base); base_w = origin + base_idx * voxel

    # hole GT = gt surface in UNKNOWN, far from observed
    d_gt_obs = cKDTree(obs_pts).query(gt_pts, workers=4)[0]
    hole = gt_pts[d_gt_obs > 3 * voxel]
    shallow = gt_pts[(d_gt_obs > 3 * voxel) & (d_gt_obs <= 25 * voxel)]
    deep = gt_pts[d_gt_obs > 25 * voxel]

    # ---- A: alpha-shape room clip ----
    alpha = 12 * voxel  # 24cm max triangle edge -> concave hull that hugs the walls
    tri, valid = alpha_shape_2d(obs_pts[:, :2], alpha)
    zlo, zhi = obs_pts[:, 2].min() - 0.05, obs_pts[:, 2].max() + 0.05
    simp = tri.find_simplex(base_w[:, :2])
    inside_room = (simp >= 0) & valid[simp] & (base_w[:, 2] >= zlo) & (base_w[:, 2] <= zhi)

    # ---- B: free-space veto (fraction of local nbhd that is FREE) ----
    freef = uniform_filter((mask == FREE).astype(np.float32), size=5)
    free_frac = freef[base_idx[:, 0], base_idx[:, 1], base_idx[:, 2]]
    free_floater = free_frac > 0.5   # >half of 5^3 nbhd is seen-empty -> floating in open air

    # ---- C: plane-preserve ----
    planes = ransac_planes(obs_pts, n_planes=6, thr=0.03)
    on_plane = np.zeros(len(base_w), bool)
    for n, d in planes:
        on_plane |= np.abs(base_w @ n + d) < 1.5 * voxel

    # ---- compose ----
    keep_abc = (inside_room & ~free_floater) | (on_plane & inside_room)
    # plain bbox clip (the v3 equivalent of what was shown)
    glo, ghi = obs_pts.min(0) - 0.05, obs_pts.max(0) + 0.05
    keep_bbox = ((base_w >= glo) & (base_w <= ghi)).all(1)
    # frontier-mask(25) reference
    keep_f25 = cKDTree(obs_pts).query(base_w, workers=4)[0] <= 25 * voxel

    # ---- scoring ----
    def rec(H, P): return 100 * np.mean(cKDTree(P).query(H, workers=4)[0] <= 2 * voxel) if len(P) and len(H) else 0.0
    def prec(P):   # frac of pred within 1.5vox of ANY true surface (obs or gt)
        if not len(P): return 0.0
        true = np.concatenate([obs_pts, gt_pts])
        return 100 * np.mean(cKDTree(true).query(P, workers=4)[0] <= 1.5 * voxel)
    def extp(P):
        return 100 * np.mean(((P < glo) | (P > ghi)).any(1)) if len(P) else 0.0

    print(f"\n=== {args.tag}  (grid {mask.shape}, surf-thr {args.surf_thr}m, {len(planes)} planes) ===")
    print(f"holes: shallow={len(shallow):,} deep={len(deep):,}   base pred voxels={len(base_w):,}")
    print(f"{'config':20s} {'shallowR':>9} {'deepR':>7} {'prec':>6} {'ext%':>6} {'nvox':>9}")
    for nm, k in [("raw", np.ones(len(base_w), bool)), ("frontier-mask(25)", keep_f25),
                  ("bbox-clip", keep_bbox), ("A+B+C", keep_abc)]:
        P = base_w[k]
        print(f"{nm:20s} {rec(shallow,P):8.1f}% {rec(deep,P):6.1f}% {prec(P):5.1f}% {extp(P):6.1f} {len(P):9d}")

    # ---- export plys ----
    import os; os.makedirs(args.out_dir, exist_ok=True)
    def wply(path, W, extra_free=None, extra_plane=None):
        d = cKDTree(obs_pts).query(W, workers=4)[0]
        col = np.where((d > 0.06)[:, None], np.array([[40, 220, 40]]), np.array([[170, 170, 170]]))
        with open(path, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(W)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, cc in zip(W, col):
                f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {int(cc[0])} {int(cc[1])} {int(cc[2])}\n")
    wply(f"{args.out_dir}/{args.tag}_bboxclip.ply", base_w[keep_bbox])
    wply(f"{args.out_dir}/{args.tag}_abc.ply", base_w[keep_abc])
    print(f"wrote {args.out_dir}/{args.tag}_bboxclip.ply  n={int(keep_bbox.sum()):,}")
    print(f"wrote {args.out_dir}/{args.tag}_abc.ply       n={int(keep_abc.sum()):,}")


if __name__ == "__main__":
    main()
