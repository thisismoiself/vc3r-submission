#!/usr/bin/env python3
"""Replica cache whose PARTIAL/INPUT is the REAL DA3 reconstruction (with its holes+noise), not a
clean-mesh visibility carve. This is the actual deployment task: DA3 reconstructs the scene from
RGB, DiffComplete fills DA3's holes. GT stays the clean Replica mesh.

Pipeline per room:
  * run DA3 (LARGE-1.1) on the trajectory frames with GT extrinsics/intrinsics -> predicted depth
  * confidence-filter, back-project to WORLD points (the DA3 reconstruction)
  * feed DA3 depth (low-conf -> inf, i.e. holes) + world points as ext_* to precompute_from_mesh,
    which carves observed/free/UNKNOWN from DA3 (unknown = DA3 holes to fill) and builds the partial
    TSDF from DA3 points, while GT TSDF comes from the clean mesh.
band=0.10 to match the current model. Held-out rooms use the full trajectory (honest deployment eval)."""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import trimesh

REPO = Path("/usr/prakt/s0016/vc3r")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "da3" / "src"))
sys.path.insert(0, str(REPO / "experiments" / "overfit_8frames"))
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
import types
sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
from tsdf_dataset import precompute_from_mesh

ROOT = Path("/usr/prakt/s0016/Replica")   # overridable via --root (e.g. NeuralRGBD, same layout)


def load_K():
    p = json.load(open(ROOT / "cam_params.json"))["camera"]
    return np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], np.float32)


def run_da3(model, room, fids, K, conf_pct, process_res, chunk):
    """DA3 depth on the given frames (GT extrinsics) -> per-frame depth maps (low-conf=inf) + world pts."""
    results = ROOT / room / "results"
    poses_all = np.loadtxt(ROOT / room / "traj.txt", np.float64).reshape(-1, 4, 4)   # c2w OpenCV RAW
    depths_out, world_all, Kp0 = [], [], None
    for s in range(0, len(fids), chunk):
        sub = fids[s:s + chunk]
        paths = [str(results / f"frame{f:06d}.jpg") for f in sub]
        c2w = poses_all[sub].astype(np.float32); w2c = np.linalg.inv(c2w)
        Kin = np.tile(K[None], (len(sub), 1, 1))
        with torch.no_grad():
            p = model.inference(image=paths, extrinsics=w2c, intrinsics=Kin,
                                align_to_input_ext_scale=True, process_res=process_res)
        depth = np.asarray(p.depth); conf = None if p.conf is None else np.asarray(p.conf)
        Kp = np.asarray(p.intrinsics); Kp0 = Kp[0] if Kp0 is None else Kp0
        H, W = depth.shape[-2:]
        cthr = np.percentile(conf, conf_pct) if conf is not None else 0.0
        world, _ = _depths_to_world_points_with_colors(
            depth, Kp, w2c, np.zeros((len(sub), H, W, 3), np.uint8), conf, cthr)
        world = world[np.isfinite(world).all(1)]
        world_all.append(world)
        for i in range(len(sub)):                                  # low-conf depth -> inf (= a hole)
            d = depth[i].astype(np.float32).copy()
            if conf is not None:
                d[conf[i] < cthr] = np.inf
            depths_out.append(d)
    return depths_out, np.concatenate(world_all, 0), Kp0


def process(model, room, out_dir, n_frames, conf_pct, process_res, chunk, band=0.10):
    out = Path(out_dir) / f"replica_{room}.npz"
    if out.exists():
        print(f"[skip] {room}"); return
    tm = trimesh.load(str(ROOT / f"{room}_mesh.ply"), process=False, force="mesh")
    verts, faces = np.asarray(tm.vertices), np.asarray(tm.faces)
    poses_all = np.loadtxt(ROOT / room / "traj.txt", np.float64).reshape(-1, 4, 4)
    fids = np.unique(np.linspace(0, len(poses_all) - 1, n_frames).round().astype(int))
    K = load_K()
    depths, world, Kp = run_da3(model, room, list(fids), K, conf_pct, process_res, chunk)
    poses_sub = poses_all[fids]
    lo = verts.min(0) - 0.3; hi = verts.max(0) + 0.3
    print(f"[{room}] DA3 pts={len(world):,}  frames={len(depths)}  -> carving", flush=True)
    precompute_from_mesh(verts, faces, poses_sub, K, 0, 0, str(out), n_frames=len(poses_sub),
                         name=room, kdt_workers=4, bounds=(lo, hi), max_vox=60_000_000, band=band,
                         ext_depths=depths, ext_K=Kp, ext_partial=world)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=["office0", "office1", "office2", "office3",
                                                   "office4", "room0", "room1", "room2"])
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_replica_da3")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--conf-pct", type=float, default=40.0)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--chunk", type=int, default=12, help="frames per DA3 inference call (VRAM)")
    ap.add_argument("--band", type=float, default=0.10, help="TSDF truncation band (m); use 0.06 to match the v3/DA3-sim models")
    ap.add_argument("--root", default=None, help="dataset root (Replica-layout); default Replica. e.g. /usr/prakt/s0016/NeuralRGBD")
    args = ap.parse_args()
    if args.root:
        global ROOT
        ROOT = Path(args.root)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to("cuda").eval()
    for r in args.rooms:
        try:
            process(model, r, args.out_dir, args.n_frames, args.conf_pct, args.process_res, args.chunk, args.band)
        except Exception as e:
            print(f"[{r}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
