#!/usr/bin/env python3
"""Presentable surface mesh of a whole-room DiffComplete completion: observed surface + predicted
hole-fills, bbox-clipped (no exterior), smooth marching-cubes surface, coloured grey(observed)/
green(filled) with baked lambert shading so form reads in a flat viewer."""
import sys, argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.measure import marching_cubes
OBS, FREE, UNK = 0, 1, 2


def build(udfpred, cache, surf_thr=None, sigma=0.6):
    u = np.load(udfpred)["udf_pred"].astype(np.float32)
    c = np.load(cache)
    mask = c["mask"]; gt = c["gt_tsdf"].astype(np.float32); pt = c["partial_tsdf"].astype(np.float32)
    origin = c["origin"].astype(np.float64); voxel = float(c["voxel"]); band = float(c["band"])
    surf = surf_thr if surf_thr else 0.9 * voxel

    # combined completion field: observed surface where OBSERVED, prediction where UNKNOWN, empty else
    field = np.full(mask.shape, band, np.float32)
    field[mask == OBS] = np.abs(pt[mask == OBS])
    field[mask == UNK] = u[mask == UNK]
    # ORIENTED bbox clip: the room may be rotated in the world frame, so an axis-aligned box keeps
    # corner hallucination. Fit the room rotation from the observed XY footprint (PCA) and clip the
    # occupied voxels in that rotated frame (Z stays world-vertical).
    obs_idx = np.argwhere((mask == OBS) & (np.abs(pt) < 0.7 * voxel))
    obs_w = origin + obs_idx * voxel
    xy = obs_w[:, :2]; ctr = xy.mean(0)
    # room orientation = MIN-AREA oriented rectangle of the footprint (robust; PCA mis-fits near-square
    # rooms like office4 -> spurious ~47deg, box bigger than AABB). Brute force over 0-90deg on the hull.
    try:
        from scipy.spatial import ConvexHull
        _hp = (xy - ctr)[ConvexHull(xy - ctr).vertices]
    except Exception:
        _hp = xy - ctr
    _best = (float("inf"), 0.0)
    for _a in np.deg2rad(np.arange(0, 90, 0.5)):
        _R = np.array([[np.cos(_a), -np.sin(_a)], [np.sin(_a), np.cos(_a)]])
        _ar = float(np.prod((_hp @ _R).max(0) - (_hp @ _R).min(0)))
        if _ar < _best[0]:
            _best = (_ar, _a)
    _a = _best[1]
    Vr = np.array([[np.cos(_a), -np.sin(_a)], [np.sin(_a), np.cos(_a)]])
    occ_idx = np.argwhere(field < surf)
    occ_w = origin + occ_idx * voxel
    q = occ_w.copy(); q[:, :2] = (occ_w[:, :2] - ctr) @ Vr
    obr = obs_w[:, :2] @ Vr if False else obs_w.copy(); obr[:, :2] = (obs_w[:, :2] - ctr) @ Vr
    lo = obr.min(0) - 0.06; hi = obr.max(0) + 0.06
    outside = ~((q >= lo) & (q <= hi)).all(1)
    oi = occ_idx[outside]
    field[oi[:, 0], oi[:, 1], oi[:, 2]] = band

    # occupancy -> mild gaussian smooth -> iso-surface at a LOW level so thin DA3-observed shells are
    # not eroded and the surface stays AT the observed location (no outward offset). gauss0.6+level0.4
    # verified to keep 99.8% of observed surface and 100% of fills (plain gauss0.9/level0.5 was eroding
    # ~14% of thin observed walls; dilating instead moved the surface outward).
    occ = (field < surf).astype(np.float32)
    occ = gaussian_filter(occ, sigma=sigma)
    verts, faces, normals, _ = marching_cubes(occ, level=0.4)
    world = origin + verts * voxel

    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world),
                                  o3d.utility.Vector3iVector(faces))
    # decimate to keep the file compact while preserving the surface
    target = min(len(faces), 700_000)
    m = m.simplify_quadric_decimation(target)
    m.compute_vertex_normals()
    world = np.asarray(m.vertices); n = np.asarray(m.vertex_normals)

    # colour: grey if near observed, green if it's a fill (far from observed input)
    obs_pts = origin + obs_idx * voxel
    d = cKDTree(obs_pts).query(world, workers=4)[0]
    is_fill = d > 0.06
    base = np.where(is_fill[:, None], np.array([[0.30, 0.72, 0.35]]), np.array([[0.62, 0.64, 0.68]]))
    L = np.array([0.4, 0.5, 0.75]); L = L / np.linalg.norm(L)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    shade = 0.45 + 0.55 * np.clip(np.abs(n @ L), 0, 1)
    col = np.clip(base * shade[:, None], 0, 1)
    m.vertex_colors = o3d.utility.Vector3dVector(col)
    return m, int(is_fill.sum())


def write_ply(path, m):
    import open3d as o3d
    o3d.io.write_triangle_mesh(path, m, write_ascii=False, compressed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--stitch-dir", default="outputs/fullroom_stitch_v3_da3")
    ap.add_argument("--cache-dir", default="outputs/tsdf_cache_replica_da3_006")
    ap.add_argument("--out-dir", default="experiments/overfit_8frames/coarse_pcd")
    args = ap.parse_args()
    m, nfill = build(f"{args.stitch_dir}/replica_{args.room}_udfpred.npz",
                     f"{args.cache_dir}/replica_{args.room}.npz")
    out = f"{args.out_dir}/{args.room}_v3_completion.ply"
    write_ply(out, m)
    print(f"{args.room}: verts={len(m.vertices):,} faces={len(m.triangles):,} fill-verts={nfill:,} -> {out}")


if __name__ == "__main__":
    main()
