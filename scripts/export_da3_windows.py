"""Add the DA3-depth reference cloud for the three single windows already exported
(office4 val, office0 train, breakfast_room NRGBD). DA3 = Depth-Anything-3 predicted
depth (GT-FREE), unprojected with the GT poses, placed in the SAME metric first-camera
frame as the pred/oracle/GT clouds so it overlays directly. Dense (confidence-filtered,
no FPS subsample). Also reports DA3->GT accuracy next to pred/oracle."""
import sys, os, json, types, numpy as np, torch, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
REPO = Path("/usr/prakt/s0016/vc3r"); os.chdir(REPO)
for _p in ["nova3r_lib/third_party", "nova3r_lib", "da3/src", "scripts"]:
    sys.path.insert(0, _p)
os.environ.setdefault("NOVA3R_DIR", "nova3r_lib")
from cache_consecutive_windows import world_to_first_camera
sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
DEV = torch.device("cuda")
CONF_PCT = float(os.environ.get("DA3_CONF_PCT", "40.0"))   # 0 = keep all visible pixels
PROCESS_RES = 504

def astensor(x): return x.float() if torch.is_tensor(x) else torch.tensor(np.array(x), dtype=torch.float32)

print("[load] DA3 depth model")
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1").to(DEV).eval()

def native_cam(data_root):
    cam = json.load(open(Path(data_root) / "cam_params.json"))["camera"]
    K = np.array([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]], np.float32)
    return K, int(cam["h"]), int(cam["w"])

@torch.no_grad()
def da3_dense(data_root, room, fids, poses_c2w, first_c2w):
    K_native, _, _ = native_cam(data_root)
    results = Path(data_root) / room / "results"
    paths = [str(results / f"frame{f:06d}.jpg") for f in fids]
    c2w = astensor(poses_c2w).numpy().astype(np.float32); w2c = np.linalg.inv(c2w)
    Kin = np.tile(K_native[None], (len(fids), 1, 1))
    p = model.inference(image=paths, extrinsics=w2c, intrinsics=Kin,
                        align_to_input_ext_scale=True, process_res=PROCESS_RES)
    depth = np.asarray(p.depth); conf = None if p.conf is None else np.asarray(p.conf)
    Kp = np.asarray(p.intrinsics); H, W = depth.shape[-2:]
    cthr = np.percentile(conf, CONF_PCT) if conf is not None else 0.0
    da3w, _ = _depths_to_world_points_with_colors(
        depth, Kp, w2c, np.zeros((len(fids), H, W, 3), np.uint8), conf, cthr)
    da3w = da3w[np.isfinite(da3w).all(1)]
    first_cam = world_to_first_camera(torch.from_numpy(da3w).float(), astensor(first_c2w)).numpy()
    return first_cam    # metric, first-camera frame (overlays pred/oracle/GT)

JOBS = [
    ("VAL office4",   "/usr/prakt/s0016/Replica",
     REPO/"scripts/data/fc_nf16_span24_100_l13_complete/office4/1028_1050_12f_28s",
     REPO/"outputs/replica/single_window_complete_val_office4"),
    ("TRAIN office0", "/usr/prakt/s0016/Replica",
     REPO/"scripts/data/fc_nf16_span24_100_l13_complete/office0/1085_1123_14f_41s",
     REPO/"outputs/replica/single_window_complete_train_office0"),
    ("NRGBD breakfast_room", "/usr/prakt/s0016/NeuralRGBD",
     REPO/"scripts/data/nrgbd_nf16_span24_100_l13_complete/breakfast_room/1000_1077_14f_97s",
     REPO/"outputs/replica/single_window_nrgbd_complete_breakfast_room"),
]

for tag, data_root, win, out_dir in JOBS:
    win, out_dir = Path(win), Path(out_dir)
    meta = torch.load(win/"meta.pt", weights_only=False)
    room, fids = meta["room"], meta["frame_ids"]
    da3pts = da3_dense(data_root, room, fids, meta["poses_c2w"], meta["first_c2w"])
    outp = out_dir / f"{room}_{win.name}_da3.ply"
    trimesh.PointCloud(da3pts).export(outp)
    # DA3->GT accuracy against the already-exported complete-frustum GT
    gt_ply = out_dir / f"{room}_{win.name}_gt_complete.ply"
    msg = f"{len(da3pts):,} pts"
    if gt_ply.exists():
        gt = np.asarray(trimesh.load(str(gt_ply), process=False).vertices, np.float32)
        rng = np.random.default_rng(0)
        a = da3pts[rng.choice(len(da3pts), min(200_000, len(da3pts)), replace=False)]
        g = gt[rng.choice(len(gt), min(200_000, len(gt)), replace=False)]
        acc = cKDTree(g).query(a, k=1, workers=4)[0].mean()*100
        comp = cKDTree(a).query(g, k=1, workers=4)[0].mean()*100
        msg += f" | DA3->GT acc={acc:.2f}cm comp={comp:.2f}cm (200k-sub)"
    print(f"[{tag}] {room}/{win.name}: {msg} -> {outp.name}", flush=True)
print("DONE")
