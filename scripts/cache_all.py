#!/usr/bin/env python3
"""Precompute full-scene TSDF caches for every available scene (SCRREAM + Replica office4).
Skips scenes already cached, so it's safe to re-run and picks up late-downloaded scenes."""
import sys, glob
from pathlib import Path
sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import precompute_scene

CACHE = Path("/usr/prakt/s0016/vc3r/outputs/tsdf_cache"); CACHE.mkdir(parents=True, exist_ok=True)
jobs = []
# SCRREAM: whichever scenes have a merged mesh
for m in sorted(glob.glob("/usr/prakt/s0016/SCRREAM/dataset/scene*/scene*_mesh.ply")):
    s = Path(m).parent.name
    jobs.append((m, f"/usr/prakt/s0016/SCRREAM/dataset/{s}/{s}_full_00/camera_pose",
                 f"/usr/prakt/s0016/SCRREAM/dataset/{s}/{s}_full_00/intrinsics.txt", s))
# Replica office4 (cross-dataset val)
jobs.append(("/usr/prakt/s0016/Replica/office4_mesh.ply", "/usr/prakt/s0016/Replica/office4/traj.txt",
             "/usr/prakt/s0016/Replica/cam_params.json", "office4"))

for mesh, poses, intr, stem in jobs:
    out = CACHE / f"{stem}.npz"
    if out.exists():
        print(f"[cache] {stem} exists, skip", flush=True); continue
    try:
        precompute_scene(mesh, poses, intr, str(out))
    except Exception as e:
        print(f"[cache] {stem} FAILED: {e}", flush=True)
print("[cache] all done", flush=True)
