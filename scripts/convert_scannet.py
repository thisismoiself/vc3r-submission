"""Convert one ScanNet v2 scan to the Replica layout so the existing pipeline
(stitch_office4.py --data-root ... --complete-target) can evaluate it zero-shot.

ScanNet has separate color (1296x968) and depth (640x480) cameras, but their NORMALISED
intrinsics are near-identical (fx/W 0.903 vs 0.893, cx/W ~0.499 both) and FOV ~58deg, so we
use the DEPTH camera as THE camera: depth drives geometry (crop_visible), and color frames
are resized to 640x480 for DA3 (a <1% intrinsic approximation). Filters invalid poses,
subsamples frames, and verifies the pose convention by NN-to-mesh distance."""
import sys, json, argparse, numpy as np, torch, trimesh
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0, "/usr/prakt/s0016/vc3r/da3/src")
from vc3r.replica import crop_visible_world_points

ap = argparse.ArgumentParser()
ap.add_argument("--scan", default="/storage/group/dataset_mirrors/scannet/scans/scene0000_00")
ap.add_argument("--out-root", default="/usr/prakt/s0016/ScanNet")
ap.add_argument("--frame-stride", type=int, default=3, help="subsample every Nth valid frame")
args = ap.parse_args()
scan = Path(args.scan); scene = scan.name
out_root = Path(args.out_root); sdir = out_root / scene; res = sdir / "results"
res.mkdir(parents=True, exist_ok=True)

# depth intrinsic (the reference camera) + depth scale
Kd = np.loadtxt(scan / "intrinsic/intrinsic_depth.txt")[:3, :3]
DW, DH, SCALE = 640, 480, 1000.0
cam = {"fx": float(Kd[0,0]), "fy": float(Kd[1,1]), "cx": float(Kd[0,2]), "cy": float(Kd[1,2]),
       "w": DW, "h": DH, "scale": SCALE}
json.dump({"camera": cam}, open(out_root / "cam_params.json", "w"), indent=2)
print(f"[cam] depth intrinsic fx={cam['fx']:.1f} cx={cam['cx']:.1f} {DW}x{DH} scale={SCALE}")

# collect valid frames (finite pose), subsample, renumber
all_idx = sorted(int(p.stem) for p in (scan / "pose").glob("*.txt"))
valid = []
for i in all_idx:
    P = np.loadtxt(scan / f"pose/{i}.txt")
    if np.isfinite(P).all():
        valid.append((i, P))
valid = valid[::args.frame_stride]
print(f"[frames] {len(all_idx)} total -> {len(valid)} valid&subsampled (stride {args.frame_stride})")

poses = []
for new_i, (old_i, P) in enumerate(valid):
    # color -> resize to depth res
    img = Image.open(scan / f"color/{old_i}.jpg").convert("RGB").resize((DW, DH), Image.BILINEAR)
    img.save(res / f"frame{new_i:06d}.jpg", quality=95)
    # depth (uint16 mm) copy as-is
    d = np.asarray(Image.open(scan / f"depth/{old_i}.png"))
    Image.fromarray(d.astype(np.uint16)).save(res / f"depth{new_i:06d}.png")
    poses.append(P.reshape(-1))
np.savetxt(sdir / "traj.txt", np.stack(poses), fmt="%.6f")

# mesh (real reconstructed geometry) -> {scene}_mesh.ply
mesh_src = scan / f"{scene}_vh_clean_2.ply"
mesh = trimesh.load(str(mesh_src), force="mesh", process=False)
mesh.export(out_root / f"{scene}_mesh.ply")
print(f"[mesh] {mesh_src.name} -> {scene}_mesh.ply ({len(mesh.vertices):,} verts)")

# ---- verify pose convention: crop_visible over a few frames should land ON the mesh ----
mp = torch.from_numpy(trimesh.sample.sample_surface(mesh, 1_000_000)[0].astype(np.float32))
K = torch.tensor([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]])
tree = cKDTree(mp.numpy()); errs = []
for new_i in np.linspace(0, len(poses)-1, 5).astype(int):
    c2w = torch.tensor(poses[new_i].reshape(4,4), dtype=torch.float32)
    depth = torch.from_numpy(np.asarray(Image.open(res/f"depth{new_i:06d}.png"), np.float32)/SCALE)
    vis = crop_visible_world_points(points_world=mp, camera_to_world=c2w, intrinsics=K, depth=depth, depth_tolerance=0.05)
    pw = vis["points_world"].numpy()
    if len(pw): errs.append(tree.query(pw[np.random.default_rng(0).choice(len(pw), min(5000,len(pw)), replace=False)])[0].mean())
m = float(np.mean(errs))*100 if errs else float("nan")
print(f"[verify] visible-crop NN-to-mesh = {m:.2f} cm over 5 frames "
      f"({'OK (OpenCV c2w, no flip)' if m < 8 else 'BAD -> wrong pose convention (try flip)'})")
print(f"[done] -> {sdir}  ({len(poses)} frames)")
