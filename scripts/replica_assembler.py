#!/usr/bin/env python3
"""Replica -> cached TSDF grids for EVALUATION (the real deployment target). Uses the clean, complete
Replica mesh as GT and renders depth along the real traj.txt camera path to carve observed/free/
unknown + partial TSDF — identical format to Stage A/B training. Replica traj.txt poses are
camera-to-world OpenCV (confirmed by the proven da3 project_world_points), so they feed RAW, no flip."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import open3d as o3d

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import precompute_from_mesh

ROOT = Path("/usr/prakt/s0016/Replica")


def load_K(render_w):
    p = json.load(open(ROOT / "cam_params.json"))["camera"]
    W0, H0, fx, fy, cx, cy = p["w"], p["h"], p["fx"], p["fy"], p["cx"], p["cy"]
    s = render_w / W0
    K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]], np.float64)
    return K, int(round(W0 * s)), int(round(H0 * s))


def process(room, out_dir, n_frames=60, render_w=600, max_vox=60_000_000):
    out = Path(out_dir) / f"replica_{room}.npz"
    if out.exists():
        print(f"[skip] {room}"); return 1
    import trimesh                                                # Replica meshes are QUADS; open3d's
    tm = trimesh.load(str(ROOT / f"{room}_mesh.ply"), process=False, force="mesh")  # PLY reader aborts
    verts, faces = np.asarray(tm.vertices), np.asarray(tm.faces)  # on them -> trimesh triangulates
    poses = np.loadtxt(ROOT / room / "traj.txt", np.float64).reshape(-1, 4, 4)   # c2w OpenCV, RAW
    idx = np.linspace(0, len(poses) - 1, n_frames).round().astype(int)
    poses = poses[idx]
    K, W, H = load_K(render_w)
    lo = verts.min(0) - 0.3; hi = verts.max(0) + 0.3
    dims = np.ceil((hi - lo) / 0.02).astype(int)
    print(f"[{room}] V={len(verts):,} F={len(faces):,} grid~{list(dims)} ({np.prod(dims)/1e6:.1f}M)", flush=True)
    precompute_from_mesh(verts, faces, poses, K, W, H, str(out), n_frames=n_frames,
                         name=room, kdt_workers=4, bounds=(lo, hi), max_vox=max_vox)
    return 1 if out.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=["office0", "office1", "office2", "office3",
                                                   "office4", "room0", "room1", "room2"])
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_replica")
    ap.add_argument("--n-frames", type=int, default=60)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for r in args.rooms:
        try:
            process(r, args.out_dir, args.n_frames)
        except Exception as e:
            print(f"[{r}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
