#!/usr/bin/env python3
"""Trajectory augmentation for Replica Stage-B fine-tuning: carve each TRAIN room from several
different CONTIGUOUS SEGMENTS of its real traj.txt (as if the camera only walked through part of the
room). Each segment leaves different regions unobserved -> different occluded/target patterns from
the SAME clean geometry, multiplying crop diversity without new scenes. Held-out rooms are NOT
augmented (their single full-trajectory cache is the honest eval). band=0.10 to match the model."""
import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import precompute_from_mesh
from replica_assembler import ROOT, load_K

# contiguous fractional windows of the trajectory -> partial room coverage (start_frac, end_frac)
SEGMENTS = [(0.00, 0.55), (0.20, 0.75), (0.45, 1.00), (0.00, 0.40), (0.60, 1.00)]
TRAIN_ROOMS = ["office0", "office1", "office2", "room0", "room1"]


def augment(room, out_dir, n_frames=45, render_w=600, max_vox=60_000_000):
    tm = trimesh.load(str(ROOT / f"{room}_mesh.ply"), process=False, force="mesh")
    verts, faces = np.asarray(tm.vertices), np.asarray(tm.faces)
    allposes = np.loadtxt(ROOT / room / "traj.txt", np.float64).reshape(-1, 4, 4)   # c2w OpenCV RAW
    K, W, H = load_K(render_w)
    lo = verts.min(0) - 0.3; hi = verts.max(0) + 0.3
    for k, (a, b) in enumerate(SEGMENTS):
        out = Path(out_dir) / f"replica_{room}_seg{k}.npz"
        if out.exists():
            print(f"[skip] {out.name}"); continue
        i0, i1 = int(a * len(allposes)), int(b * len(allposes))
        seg = allposes[i0:i1]
        idx = np.linspace(0, len(seg) - 1, min(n_frames, len(seg))).round().astype(int)
        poses = seg[idx]
        print(f"[{room} seg{k}] frames {i0}-{i1} -> {len(poses)} poses", flush=True)
        precompute_from_mesh(verts, faces, poses, K, W, H, str(out), n_frames=len(poses),
                             name=f"{room}_seg{k}", kdt_workers=4, bounds=(lo, hi), max_vox=max_vox)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/usr/prakt/s0016/vc3r/outputs/tsdf_cache_replica_ft_train"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for r in TRAIN_ROOMS:
        try:
            augment(r, out_dir)
        except Exception as e:
            print(f"[{r}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
