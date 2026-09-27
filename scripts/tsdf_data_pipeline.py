#!/usr/bin/env python3
"""TSDF hole-filling DATA PIPELINE (Stage-0 of tsdf.md) + visualization.

Takes a complete mesh + camera trajectory and produces, on a dense voxel crop:
  - partial point cloud  (mesh depth rendered along the trajectory, back-projected)
  - observed / free / unknown voxel mask  (visibility carving)
  - partial TSDF          (signed distance to the partial cloud, normal-signed, truncated)
  - GT TSDF               (signed distance to the complete mesh, truncated)
Supervision at train time is on `unknown` voxels only. This script just BUILDS and
VISUALIZES them (the spec's step 1: "visualize the masks and both TSDFs before training").

Generic in (mesh, poses, intrinsics) so SCRREAM/synthetic meshes drop in for real training.
Runs on CPU (Open3D raycasting). Bootstrapped here on Replica office4.
"""
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/usr/prakt/s0016/vc3r")


def load_poses(path):
    """Replica: a single traj.txt (Nx16). SCRREAM: a dir of per-frame NNNNNN.txt (4x4)."""
    path = Path(path)
    if path.is_dir():
        return np.stack([np.loadtxt(f, dtype=np.float64).reshape(4, 4)
                         for f in sorted(path.glob("*.txt"))])
    return np.loadtxt(path, dtype=np.float64).reshape(-1, 4, 4)


def load_intrinsics(path):
    """cam_params.json (Replica) or a 3x3 intrinsics.txt (SCRREAM). Returns K, W, H."""
    path = Path(path)
    if path.suffix == ".json":
        c = __import__("json").load(open(path))["camera"]
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
        return K, int(c["w"]), int(c["h"])
    K = np.loadtxt(path, dtype=np.float64)
    return K, int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))  # principal point centered


def render_depth(scene, K, c2w, W, H):
    """Cam-z depth by raycasting the mesh (OpenCV convention: +z forward). Returns depth[H,W],
    and the world hit points for finite hits."""
    extr = np.linalg.inv(c2w)                                    # world->cam
    rays = scene.create_rays_pinhole(o3d.core.Tensor(K, o3d.core.float32),
                                     o3d.core.Tensor(extr, o3d.core.float32), W, H)
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()                                 # [H,W] range along (normalized) dir
    rd = rays.numpy()                                            # [H,W,6] origin+dir
    finite = np.isfinite(t_hit)
    hit_world = rd[..., :3] + t_hit[..., None] * rd[..., 3:]     # [H,W,3]
    # cam-z depth = (hit_world - C) . forward_axis
    R = c2w[:3, :3]; C = c2w[:3, 3]
    fwd = R[:, 2]                                                # +z camera axis in world
    depth = ((hit_world - C) @ fwd)
    depth[~finite] = np.inf
    return depth, hit_world[finite]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="office4")
    ap.add_argument("--data-root", default="/usr/prakt/s0016/Replica")
    ap.add_argument("--mesh", default=None, help="explicit scene mesh (else Replica {room}_mesh.ply)")
    ap.add_argument("--poses", default=None, help="traj.txt (Replica) or camera_pose/ dir (SCRREAM)")
    ap.add_argument("--intrinsics", default=None, help="cam_params.json or 3x3 intrinsics.txt")
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(0, 71, 10)))  # one 8-frame window
    ap.add_argument("--voxel", type=float, default=0.02)        # 2 cm
    ap.add_argument("--grid", type=int, default=128)            # 128^3 -> 2.56 m
    ap.add_argument("--band-vox", type=float, default=5.0)      # truncation = 5 voxels = 10 cm
    ap.add_argument("--out", default=str(REPO / "outputs/tsdf_pipeline/office4_win0"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    K, W, H = load_intrinsics(args.intrinsics or (root / "cam_params.json"))
    band = args.band_vox * args.voxel

    # ---- mesh + raycasting scene (load via trimesh: Open3D's PLY reader chokes on
    #      non-triangle faces; trimesh triangulates robustly; skip_materials avoids texture OOM) ----
    mesh_path = args.mesh or (root / f"{args.room}_mesh.ply")
    tm = trimesh.load(str(mesh_path), force="mesh", process=False, skip_materials=True)
    verts = np.asarray(tm.vertices, np.float32); faces = np.asarray(tm.faces, np.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(faces))
    poses = load_poses(args.poses or (root / args.room / "traj.txt"))
    print(f"[mesh] {Path(mesh_path).name}  verts={len(verts)}  faces={len(faces)}  "
          f"img={W}x{H}  poses={len(poses)}  frames={args.frames}")

    # ---- render partial cloud from the window's frames ----
    depths, cam_centers = [], []
    partial_pts = []
    for f in args.frames:
        d, hits = render_depth(scene, K, poses[f], W, H)
        depths.append(d); cam_centers.append(poses[f][:3, 3]); partial_pts.append(hits)
    partial = np.concatenate(partial_pts, 0).astype(np.float64)
    # subsample partial for speed of KD-tree / normals
    if len(partial) > 400000:
        partial = partial[np.random.default_rng(0).choice(len(partial), 400000, replace=False)]
    print(f"[partial] {len(partial):,} points from {len(args.frames)} rendered views")

    # ---- crop centred on the observed geometry (where occlusion lives) ----
    center = np.median(partial, 0)
    N = args.grid; half = N * args.voxel / 2
    lin = (np.arange(N) + 0.5) * args.voxel - half
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    vox = np.stack([gx, gy, gz], -1) + center                   # [N,N,N,3] world voxel centres
    vflat = vox.reshape(-1, 3)
    print(f"[grid] {N}^3 @ {args.voxel*100:.0f}cm = {N*args.voxel:.2f}m, centre={np.round(center,2)}")

    # ---- GT TSDF from the mesh ----
    gt_sdf = scene.compute_signed_distance(o3d.core.Tensor(vflat.astype(np.float32))).numpy()
    gt_tsdf = np.clip(gt_sdf, -band, band).reshape(N, N, N)

    # ---- visibility carving: observed / free / unknown ----
    OBSERVED, FREE, UNKNOWN = 0, 1, 2
    obs = np.zeros(len(vflat), bool); free = np.zeros(len(vflat), bool)
    eps = 1.5 * args.voxel
    for f, d in zip(args.frames, depths):
        c2w = poses[f]; w2c = np.linalg.inv(c2w)
        vc = (w2c[:3, :3] @ vflat.T + w2c[:3, 3:4]).T           # cam frame
        z = vc[:, 2]
        u = K[0, 0] * vc[:, 0] / z + K[0, 2]
        v = K[1, 1] * vc[:, 1] / z + K[1, 2]
        inb = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.clip(u.astype(int), 0, W - 1); vi = np.clip(v.astype(int), 0, H - 1)
        dref = d[vi, ui]
        valid = inb & np.isfinite(dref)
        obs |= valid & (np.abs(z - dref) < eps)                 # near the rendered surface
        free |= valid & (z < dref - eps)                        # ray passed through -> free
    mask = np.full(len(vflat), UNKNOWN, np.int8)
    mask[free] = FREE
    mask[obs] = OBSERVED                                        # observed overrides free
    mask = mask.reshape(N, N, N)
    nun, nob, nfr = (mask == UNKNOWN).sum(), (mask == OBSERVED).sum(), (mask == FREE).sum()
    print(f"[carve] observed={nob:,} free={nfr:,} unknown={nun:,}  (unknown {100*nun/mask.size:.1f}%)")

    # ---- partial TSDF: normal-signed distance to the partial cloud, truncated ----
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.mean(cam_centers, 0))
    pn = np.asarray(pcd.normals)
    tree = cKDTree(partial)
    dist, idx = tree.query(vflat, k=1)
    sign = np.sign(np.einsum("ij,ij->i", vflat - partial[idx], pn[idx]))
    sign[sign == 0] = 1.0
    part_tsdf = np.clip(sign * dist, -band, band).reshape(N, N, N)

    # ================= VISUALIZE =================
    def save_ply(pts, path, color=None):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if color is not None:
            p.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
        o3d.io.write_point_cloud(str(path), p)

    save_ply(partial, out / "partial_points.ply", [0.2, 0.6, 1.0])
    save_ply(vflat[mask.reshape(-1) == UNKNOWN], out / "voxels_unknown.ply", [1.0, 0.2, 0.2])
    save_ply(vflat[mask.reshape(-1) == OBSERVED], out / "voxels_observed.ply", [0.2, 0.9, 0.2])
    # GT surface = |gt_tsdf| < 1 voxel  (sanity: should reproduce the mesh in the crop)
    save_ply(vflat[np.abs(gt_tsdf.reshape(-1)) < args.voxel], out / "gt_tsdf_zeroset.ply", [0.9, 0.9, 0.2])
    # the actual training target region: unknown voxels near the GT surface
    tgt = (mask.reshape(-1) == UNKNOWN) & (np.abs(gt_tsdf.reshape(-1)) < band)
    save_ply(vflat[tgt], out / "target_unknown_surface.ply", [1.0, 0.5, 0.0])

    # 2D slices through the crop centre for the two TSDFs + mask
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    sl008 = [N // 4, N // 2, 3 * N // 4]
    cmap_mask = matplotlib.colors.ListedColormap(["#22e022", "#3399ff", "#ee2222"])  # obs/free/unknown
    for col, k in enumerate(sl008):
        axes[0, col].imshow(gt_tsdf[:, :, k], cmap="coolwarm", vmin=-band, vmax=band)
        axes[0, col].set_title(f"GT TSDF  z-slice {k}")
        axes[1, col].imshow(part_tsdf[:, :, k], cmap="coolwarm", vmin=-band, vmax=band)
        axes[1, col].set_title(f"partial TSDF  z-slice {k}")
        axes[2, col].imshow(mask[:, :, k], cmap=cmap_mask, vmin=0, vmax=2)
        axes[2, col].set_title(f"mask (green=obs blue=free red=unk) z {k}")
    for ax in axes.ravel(): ax.axis("off")
    plt.tight_layout(); plt.savefig(out / "slices.png", dpi=110); plt.close()

    stats = {"room": args.room, "frames": args.frames, "voxel_m": args.voxel, "grid": N,
             "band_m": band, "partial_points": int(len(partial)),
             "observed": int(nob), "free": int(nfr), "unknown": int(nun),
             "unknown_frac": float(nun / mask.size),
             "target_unknown_surface_voxels": int(tgt.sum()),
             "gt_tsdf_range": [float(gt_tsdf.min()), float(gt_tsdf.max())],
             "partial_tsdf_range": [float(part_tsdf.min()), float(part_tsdf.max())]}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    np.savez_compressed(out / "grid.npz", gt_tsdf=gt_tsdf.astype(np.float16),
                        partial_tsdf=part_tsdf.astype(np.float16), mask=mask.astype(np.int8),
                        center=center, voxel=args.voxel)
    print(f"[out] {out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
