#!/usr/bin/env python3
"""Assemble a complete (amodal) room mesh from a 3D-FRONT layout JSON + 3D-FUTURE furniture,
and simulate a camera trajectory inside it. This is the Stage-A training-data source: 3D-FRONT
gives professionally-designed rooms with *watertight furniture* placed by pose, so the assembled
room mesh has true occluded geometry (undersides of tables, backs of sofas) as GT.

Coordinate facts (verified from the data):
  * Up axis is +Y. Floor meshes sit at world y=0; furniture `pos` carries height.
  * Architecture meshes (Wall*, Floor, Ceiling, ...) store `xyz` already in WORLD coords.
  * Furniture instance `rot` is a quaternion in [x,y,z,w] order (scipy convention); the identity
    (0,0,0,1) dominates and all rotations are about Y.
  * Furniture placement: world = pos + R(rot) @ (scale * v_local), on 3D-FUTURE raw_model.obj.

Public API:
  iter_rooms(json_path) -> yields (room_id, verts, faces, floor_xyz) for usable rooms
  sample_trajectory(verts, faces, floor_xyz, n, rng) -> poses [N,4,4] c2w (OpenCV: +z fwd, +y down)
  default_intrinsics() -> K, W, H
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

FRONT = Path("/storage/group/dataset_mirrors/01_incoming/3DFront")
LAYOUTS = FRONT / "3D-FRONT"
MODELS = FRONT / "3D-FUTURE-model"

# architecture mesh types that are real surfaces (skip light bands / decorative slabs that
# would confuse the TSDF sign, keep the ones that bound the room)
ARCH_KEEP = {"Floor", "Ceiling", "WallInner", "WallOuter", "WallTop", "WallBottom",
             "Baseboard", "SlabBottom", "SlabSide", "SlabTop", "Door", "Window",
             "Front", "Back", "Hole", "Pocket", "CustomizedFeatureWall",
             "CustomizedFixedFurniture", "CustomizedPlatform"}

_MODEL_CACHE = {}


def _load_model(jid):
    """Load a 3D-FUTURE model's raw geometry (verts, faces), cached across instances."""
    if jid in _MODEL_CACHE:
        return _MODEL_CACHE[jid]
    obj = MODELS / jid / "raw_model.obj"
    if not obj.exists():
        _MODEL_CACHE[jid] = None
        return None
    try:
        m = trimesh.load(str(obj), force="mesh", process=False, skip_materials=True)
        v = np.asarray(m.vertices, np.float32); f = np.asarray(m.faces, np.int64)
        if len(v) == 0 or len(f) == 0:
            v = f = None
    except Exception:
        v = f = None
    _MODEL_CACHE[jid] = (v, f) if v is not None else None
    return _MODEL_CACHE[jid]


def iter_rooms(json_path, min_furn=3, min_area=3.0, max_area=80.0, skip_dir=None):
    """Yield (room_id, verts, faces, floor_xyz) for each usable furnished room in a house.
    `skip_dir`: if a room's output npz already exists there, skip it BEFORE assembling its mesh
    (furniture OBJ loads are the expensive part) — makes resumes cheap."""
    j = json.load(open(json_path))
    # lookups
    mesh_by_uid = {m["uid"]: m for m in j.get("mesh", [])}
    furn_by_uid = {f["uid"]: f for f in j.get("furniture", []) if f.get("jid")}
    house = Path(json_path).stem

    for room in j["scene"]["room"]:
        if skip_dir is not None and (Path(skip_dir) / f"{house}_{room['instanceid']}.npz").exists():
            continue                                             # already cached -> skip before assembly
        V, F, off = [], [], 0
        floor_xyz = []
        n_furn = 0
        for ch in room.get("children", []):
            ref = ch["ref"]
            if ref in mesh_by_uid:                                   # architecture piece (world coords)
                m = mesh_by_uid[ref]
                if m.get("type") not in ARCH_KEEP:
                    continue
                xyz = np.asarray(m["xyz"], np.float32).reshape(-1, 3)
                fa = np.asarray(m["faces"], np.int64).reshape(-1, 3)
                if m.get("type") == "Floor":
                    floor_xyz.append(xyz)
                V.append(xyz); F.append(fa + off); off += len(xyz)
            elif ref in furn_by_uid:                                 # furniture instance -> place model
                jid = furn_by_uid[ref]["jid"]
                mv = _load_model(jid)
                if mv is None:
                    continue
                v, fa = mv
                R = Rotation.from_quat(ch["rot"]).as_matrix().astype(np.float32)   # [x,y,z,w]
                s = np.asarray(ch["scale"], np.float32); p = np.asarray(ch["pos"], np.float32)
                vw = (R @ (v * s).T).T + p
                V.append(vw.astype(np.float32)); F.append(fa + off); off += len(vw)
                n_furn += 1
        if n_furn < min_furn or not floor_xyz:
            continue
        floor_xyz = np.concatenate(floor_xyz, 0)
        # footprint area from floor bbox in the xz-plane
        fp = floor_xyz[:, [0, 2]]
        area = (fp[:, 0].ptp()) * (fp[:, 1].ptp())
        if not (min_area <= area <= max_area):
            continue
        verts = np.concatenate(V, 0).astype(np.float32)
        faces = np.concatenate(F, 0).astype(np.uint32)
        yield f"{house}_{room['instanceid']}", verts, faces, floor_xyz


def default_intrinsics(W=640, H=480, hfov_deg=70.0):
    f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    return K, W, H


def _lookat(eye, target):
    """c2w for OpenCV convention (+z forward, +x right, +y down), matching render_depth."""
    z = target - eye; z /= (np.linalg.norm(z) + 1e-9)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))                        # world up = +Y
    if np.linalg.norm(x) < 1e-6:                                      # looking straight up/down
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)                                                # points down
    c2w = np.eye(4)
    c2w[:3, 0] = x; c2w[:3, 1] = y; c2w[:3, 2] = z; c2w[:3, 3] = eye
    return c2w


def sample_trajectory(verts, faces, floor_xyz, n=40, rng=None, cam_h=(1.1, 1.7), sweep_frac=0.5):
    """Cameras inside the room footprint at human height. Two components:
      * a COVERAGE SWEEP (sweep_frac of frames): a grid of eye positions, each panning through
        several outward yaws, so the depth rays sweep the whole room and carve free space
        thoroughly. Dense free-space carving is what stops the model hallucinating into empty
        volume and what makes the exterior->FREE flood-fill safe (a well-closed observed shell).
      * FURNITURE / interior views (the rest): random eyes looking at furniture points, which
        create the meaningful occluded (`unknown`) pockets behind objects that are the target."""
    rng = rng or np.random.default_rng(0)
    fx = floor_xyz[:, 0]; fz = floor_xyz[:, 2]
    x0, x1, z0, z1 = fx.min(), fx.max(), fz.min(), fz.max()
    floor_y = float(np.median(floor_xyz[:, 1]))
    ceil_y = float(verts[:, 1].max())
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    mx = 0.15 * (x1 - x0); mz = 0.15 * (z1 - z0)                      # keep cameras off the walls
    poses = []

    # --- coverage sweep: grid of eyes, each panning through evenly-spaced outward yaws ---
    n_sweep = int(round(n * sweep_frac))
    n_yaw = 6
    g = max(2, int(round(np.sqrt(max(1.0, n_sweep / n_yaw)))))       # grid side
    yaws = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)
    k = 0
    for ex in np.linspace(x0 + mx, x1 - mx, g):
        for ez in np.linspace(z0 + mz, z1 - mz, g):
            eye = np.array([ex, floor_y + rng.uniform(*cam_h), ez], np.float64)
            phase = rng.uniform(0, 2 * np.pi)
            for yaw in yaws + phase:
                if k >= n_sweep:
                    break
                d = np.array([np.cos(yaw), rng.uniform(-0.35, 0.15), np.sin(yaw)])  # slight down/up pan
                poses.append(_lookat(eye, eye + d)); k += 1

    # --- furniture / interior views: the occlusion-creating look-ats (old behaviour) ---
    interior = verts[(verts[:, 1] > floor_y + 0.1) & (verts[:, 1] < min(ceil_y, floor_y + 2.0))]
    for i in range(n - len(poses)):
        eye = np.array([rng.uniform(x0 + mx, x1 - mx),
                        floor_y + rng.uniform(*cam_h),
                        rng.uniform(z0 + mz, z1 - mz)], np.float64)
        if len(interior) and rng.random() < 0.7:                     # look at a furniture point
            tgt = interior[rng.integers(len(interior))].astype(np.float64)
        else:                                                        # look toward room centre
            tgt = np.array([cx + rng.uniform(-mx, mx), floor_y + rng.uniform(0.3, 1.2),
                            cz + rng.uniform(-mz, mz)], np.float64)
        if np.linalg.norm(tgt - eye) < 0.3:
            tgt = tgt + np.array([0.5, 0, 0.5])
        poses.append(_lookat(eye, tgt))
    return np.stack(poses[:n])


def process_house(hp, out_dir, n_frames, seed, band=0.10, free_flood=False, voxel=0.02):
    """Cache every usable room in one house (one multiprocessing work item)."""
    import warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
    from tsdf_dataset import precompute_from_mesh
    K, W, H = default_intrinsics()
    out = Path(out_dir); n = 0
    try:
        rooms = list(iter_rooms(hp, skip_dir=out_dir))           # cheap resume: skip cached rooms pre-assembly
    except Exception as e:
        return 0
    for room_id, verts, faces, floor_xyz in rooms:
        npz = out / f"{room_id}.npz"
        if npz.exists():
            n += 1; continue
        rng = np.random.default_rng(abs(hash(room_id)) % (2 ** 32) ^ seed)
        poses = sample_trajectory(verts, faces, floor_xyz, n=n_frames, rng=rng)
        # ROBUST grid bounds: x,z from the floor footprint (walls ~= floor extent), y from floor
        # level up to a capped room height. Prevents a single misplaced-furniture outlier vertex
        # from inflating the grid to billions of voxels (which OOMs the node).
        floor_y = float(np.median(floor_xyz[:, 1]))
        lo = np.array([floor_xyz[:, 0].min() - 0.4, floor_y - 0.15, floor_xyz[:, 2].min() - 0.4])
        hi = np.array([floor_xyz[:, 0].max() + 0.4, floor_y + 3.5, floor_xyz[:, 2].max() + 0.4])
        try:
            # one thread per worker: the multiprocessing Pool provides the parallelism, so each
            # worker's KD-tree must NOT grab all cores (that oversubscribes the node ~Nx).
            # cap grid at ~22M vox: we only ever train on 128^3 crops, so a giant full-room grid
            # is wasted compute (single-threaded it can take 10-20 min). Bigger rooms are skipped.
            precompute_from_mesh(verts, faces, poses, K, W, H, str(npz), n_frames=n_frames,
                                 name=room_id, kdt_workers=2, bounds=(lo, hi), max_vox=22_000_000,
                                 band=band, free_flood=free_flood, voxel=voxel)
            n += 1
        except Exception as e:
            print(f"[skip] {room_id}: {e}", flush=True)
    return n


def main():
    """Assemble rooms from N houses and write cached TSDF grids (reuses tsdf_dataset core)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/usr/prakt/s0016/vc3r/outputs/tsdf_cache_front3d")
    ap.add_argument("--n-houses", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-rooms", type=int, default=100000)
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1, help="parallel worker processes over houses")
    ap.add_argument("--band", type=float, default=0.10, help="TSDF truncation band in metres (0.06 = 3 vox, paper)")
    ap.add_argument("--free-flood", action="store_true", help="relabel unobserved empty space UNKNOWN->FREE (exterior/gaps)")
    ap.add_argument("--voxel", type=float, default=0.02, help="voxel size (m); 0.08 for the coarse cascade stage")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(str(LAYOUTS / "*.json")))[args.start:args.start + args.n_houses]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.list_only:
        done = 0
        for hp in files:
            try:
                rooms = list(iter_rooms(hp))
            except Exception:
                continue
            for room_id, verts, faces, floor_xyz in rooms:
                print(f"[room] {room_id} verts={len(verts)} faces={len(faces)} "
                      f"floor_bbox_area~{floor_xyz[:,[0,2]].ptp(0).prod():.1f}m2", flush=True)
                done += 1
                if done >= args.max_rooms:
                    return
        return

    if args.procs > 1:
        import multiprocessing as mp
        from functools import partial
        # 'spawn' (not the Linux default 'fork'): Open3D / OpenBLAS start internal threads at
        # import, and forking a threaded parent deadlocks the workers on the first Open3D call.
        # spawn gives each worker a fresh interpreter that imports Open3D itself.
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ctx.Pool(args.procs) as pool:
            counts = pool.map(partial(process_house, out_dir=str(out), n_frames=args.n_frames,
                                      seed=args.seed, band=args.band, free_flood=args.free_flood,
                                      voxel=args.voxel), files, chunksize=1)
        print(f"[done] {sum(counts)} rooms from {len(files)} houses in {time.time()-t0:.0f}s", flush=True)
    else:
        total = sum(process_house(hp, str(out), args.n_frames, args.seed,
                                  band=args.band, free_flood=args.free_flood, voxel=args.voxel) for hp in files)
        print(f"[done] {total} rooms", flush=True)


if __name__ == "__main__":
    main()
