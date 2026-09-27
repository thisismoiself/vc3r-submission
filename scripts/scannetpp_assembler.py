#!/usr/bin/env python3
"""Stage-B data pipeline: ScanNet++ (REAL scans) -> cached TSDF grids, reusing the exact
precompute_from_mesh machinery from Stage A. For each scene we take the high-quality laser-scan
mesh (scans/mesh_aligned_0.05.ply) as GT geometry, and render depth from it along the REAL iPhone
camera trajectory (aligned_pose in pose_intrinsic_imu.json) to carve observed/free/unknown + build
the partial TSDF. This gives REAL room geometry + REAL scan coverage/occlusion (the main sim-to-real
gap vs 3D-FRONT); swapping in real sensor depth is a later upgrade.

ScanNet++ iPhone poses are ARKit/OpenGL-style (cam looks -Z, +Y up); precompute expects OpenCV c2w
(+Z fwd, +Y down), so we right-multiply by diag(1,-1,-1,1). We auto-detect the flip by which
convention yields more valid depth (a wrong convention points cameras away -> empty partial)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import open3d as o3d

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import precompute_from_mesh

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])                      # OpenGL/ARKit c2w -> OpenCV c2w


def load_mesh(scene_dir):
    ply = Path(scene_dir) / "scans" / "mesh_aligned_0.05.ply"
    m = o3d.io.read_triangle_mesh(str(ply))
    return np.asarray(m.vertices), np.asarray(m.triangles)


def load_poses_K(scene_dir, n_frames, render_w=640):
    """Return (poses_opencv [n,4,4], K_render, W, H). Subsamples n_frames evenly across the scan."""
    j = json.load(open(Path(scene_dir) / "iphone" / "pose_intrinsic_imu.json"))
    keys = sorted(j.keys())                                  # frame_000000 ...
    idx = np.linspace(0, len(keys) - 1, n_frames).round().astype(int)
    poses_gl = np.stack([np.array(j[keys[i]]["aligned_pose"], np.float64) for i in idx])
    K_full = np.array(j[keys[idx[0]]]["intrinsic"], np.float64)
    # full-res iPhone RGB is ~1920x1440 (principal point ~ half); scale K to a render resolution
    W_full = 2 * K_full[0, 2]
    s = render_w / W_full
    W = int(round(W_full * s)); H = int(round(2 * K_full[1, 2] * s))
    K = K_full.copy(); K[:2, :] *= s
    return poses_gl, K, W, H


def _valid_frac(verts, faces, poses, K, W, H):
    """Rough check that cameras see the mesh: raycast a few poses, fraction of finite-depth pixels."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts, o3d.core.float32), o3d.core.Tensor(faces, o3d.core.uint32))
    frac = []
    for c2w in poses[:: max(1, len(poses) // 6)][:6]:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K, o3d.core.float32),
            o3d.core.Tensor(np.linalg.inv(c2w), o3d.core.float32), W, H)
        d = scene.cast_rays(rays)["t_hit"].numpy()
        frac.append(np.isfinite(d).mean())
    return float(np.mean(frac))


def process_scene(scene_dir, out_dir, n_frames=50, pose_conv="raw", max_vox=40_000_000):
    # CONVENTION PROVEN via COLMAP: ScanNet++ aligned_pose is ALREADY OpenCV c2w -> use RAW (no flip).
    # The old valid-depth-fraction "auto" heuristic is NOT discriminative (a camera inside a room sees
    # walls looking either way) and systematically mis-picked the flip; do NOT use it. See git history.
    scene_dir = Path(scene_dir); sid = scene_dir.name
    out = Path(out_dir) / f"scannetpp_{sid}.npz"
    if out.exists():
        print(f"[skip] {sid} exists"); return 1
    verts, faces = load_mesh(scene_dir)
    poses_gl, K, W, H = load_poses_K(scene_dir, n_frames)

    # pick the pose convention that actually looks at the mesh
    cand = {"opencv_flip": np.einsum("nij,jk->nik", poses_gl, GL2CV), "raw": poses_gl}
    if pose_conv == "auto":
        scored = {k: _valid_frac(verts, faces, p, K, W, H) for k, p in cand.items()}
        conv = max(scored, key=scored.get)
        print(f"[{sid}] pose-conv valid-frac {{{', '.join(f'{k}:{v:.2f}' for k,v in scored.items())}}} -> {conv}")
    else:
        conv = pose_conv
    poses = cand["opencv_flip" if conv in ("auto", "opencv_flip") else "raw"]

    lo = verts.min(0) - 0.3; hi = verts.max(0) + 0.3            # bounds from real mesh AABB
    dims = np.ceil((hi - lo) / 0.02).astype(int)
    print(f"[{sid}] mesh V={len(verts):,} F={len(faces):,}  grid~{list(dims)} ({np.prod(dims)/1e6:.1f}M vox)", flush=True)
    r = precompute_from_mesh(verts, faces, poses, K, W, H, str(out), n_frames=n_frames,
                             name=sid, kdt_workers=4, bounds=(lo, hi), max_vox=max_vox)
    return 1 if out.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp-root", default="/storage/group/dataset_mirrors/01_incoming/scannetpp/data")
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_scannetpp")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene ids; default = --split")
    ap.add_argument("--split", default=None, help="path to a split .txt of scene ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.scenes:
        scenes = args.scenes
    elif args.split:
        scenes = [l.strip() for l in open(args.split) if l.strip()]
    else:
        scenes = sorted(p.name for p in Path(args.sp_root).iterdir() if p.is_dir())
    if args.limit:
        scenes = scenes[:args.limit]
    print(f"[assembler] {len(scenes)} scenes -> {args.out_dir} ({args.procs} procs)", flush=True)

    dirs = [Path(args.sp_root) / sid for sid in scenes]
    if args.procs > 1:
        import multiprocessing as mp
        from functools import partial
        # 'spawn' not 'fork': Open3D spawns threads at import; forking a threaded parent deadlocks.
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.procs) as pool:
            counts = pool.map(partial(_safe, out_dir=args.out_dir, n_frames=args.n_frames), dirs, chunksize=1)
        done = sum(counts)
    else:
        done = sum(_safe(d, args.out_dir, args.n_frames) for d in dirs)
    print(f"[assembler] done {done}/{len(scenes)}", flush=True)


def _safe(scene_dir, out_dir, n_frames):
    import warnings; warnings.filterwarnings("ignore")
    try:
        return process_scene(scene_dir, out_dir, n_frames)
    except Exception as e:
        print(f"[{Path(scene_dir).name}] FAILED: {e}", flush=True)
        return 0


if __name__ == "__main__":
    main()
