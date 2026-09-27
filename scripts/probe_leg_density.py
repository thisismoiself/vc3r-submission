"""Is the chair-leg (thin-detail) miss a SAMPLING-RATE issue or absent from z*?
Decode the LoRA-decoder oracle of the same office0/office4 window at escalating query density and
measure GT->pred recall at tight thresholds (fine detail = legs live here). Rising recall with density
=> sampling-limited (denser/importance-sampled decode recovers legs). Saturating recall => legs are not
in z* (representation-compression limit; FT/sampling can't help). Exports the high-density cloud so the
legs can be inspected visually. Same cached z*, midpoint, chunked."""
import sys, os, numpy as np, torch, trimesh, json
from pathlib import Path
from scipy.spatial import cKDTree
REPO = Path("/usr/prakt/s0016/vc3r"); os.chdir(REPO)
for _p in ["nova3r_lib/third_party", "nova3r_lib", "da3/src", "scripts"]:
    sys.path.insert(0, _p)
os.environ.setdefault("NOVA3R_DIR", "nova3r_lib")
from demo_nova3r import load_model as load_nova3r_model
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from cache_consecutive_windows import world_to_first_camera
from vc3r.replica import project_world_points
from omegaconf import OmegaConf
DEV = torch.device("cuda")
NOVA_CKPT = REPO/"checkpoints/nova3r/scene_ae/checkpoint-last.pth"
LORA_HEAD = REPO/"outputs/consecutive_windows/diag_decoder_lora_head.pt"
CACHE = REPO/"scripts/data/fc_nf16_span24_100_l13_complete"
REPL = Path("/usr/prakt/s0016/Replica")
CHUNK, SEED, MESH_N = 50_000, 42, 4_000_000
DENSITIES = [200_000, 800_000, 2_000_000]
THRS = [0.005, 0.01, 0.02]

nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
[p.requires_grad_(False) for p in nova.parameters()]
OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
nova.pts3d_head.load_state_dict(torch.load(LORA_HEAD, map_location="cpu", weights_only=True), strict=False)
cam = json.load(open(REPL/"cam_params.json"))["camera"]
Hn, Wn = int(cam["h"]), int(cam["w"])
Kmat = torch.tensor([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]])

@torch.no_grad()
def decode_dense(tokens, pts_norm, nq):
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1//step)).to(DEV)
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova)); outs = []
    for ci, s in enumerate(range(0, nq, CHUNK)):
        n = min(CHUNK, nq - s); torch.manual_seed(SEED + ci)
        xi = torch.rand(1, n, 3, device=DEV)*2-1
        with torch.amp.autocast("cuda", enabled=False):
            sol = solver.sample(time_grid=T, x_init=xi, method="midpoint", step_size=step,
                return_intermediates=False, images=torch.zeros(1,1,3,1,1,device=DEV),
                token_mask=None, encoder_data={"tokens": tokens.to(DEV)}, pointmaps=pts_norm.to(DEV))
        outs.append((sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()); torch.cuda.empty_cache()
    return np.concatenate(outs, 0)

def frustum(pts, c2w, hw):
    proj = project_world_points(points_world=pts, world_to_camera=torch.linalg.inv(c2w), intrinsics=Kmat, image_hw=hw)
    return pts[proj["inside"]]
def astensor(x): return x.float() if torch.is_tensor(x) else torch.tensor(np.array(x), dtype=torch.float32)

def probe(win, out_dir, tag):
    win, out_dir = Path(win), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = torch.load(win/"meta.pt", weights_only=False)
    zmean = torch.load(win/"z_star_online_mean.pt", weights_only=False).float()
    pts_norm = torch.load(win/"pts_norm.pt", weights_only=False).float()
    nf = float(meta["norm_factor"]); room = meta["room"]; fids = meta["frame_ids"]
    poses = meta["poses_c2w"]; first = astensor(meta["first_c2w"]); scale = nf/3.0
    mesh = trimesh.load(str(REPL/f"{room}_mesh.ply"), force="mesh", process=False)
    mp = torch.from_numpy(trimesh.sample.sample_surface(mesh, MESH_N)[0].astype(np.float32))
    gt = torch.cat([world_to_first_camera(frustum(mp, astensor(poses[i]), (Hn, Wn)), first)
                    for i in range(len(fids))], 0)
    gt = torch.unique(torch.round(gt*1e4)/1e4, dim=0).numpy()
    print(f"[{tag} {room}] {win.name}  GT {len(gt):,} pts", flush=True)
    for nq in DENSITIES:
        cloud = decode_dense(zmean, pts_norm, nq) * scale
        d_gp = cKDTree(cloud).query(gt, k=1)[0]     # GT -> pred (completeness / recall)
        rec = "  ".join(f"rec@{int(t*1000)}mm={ (d_gp<t).mean()*100:5.1f}%" for t in THRS)
        print(f"    q={nq:>9,}  comp={d_gp.mean()*100:5.2f}cm  {rec}", flush=True)
        if nq == DENSITIES[-1]:
            trimesh.PointCloud(cloud).export(out_dir/f"{room}_{win.name}_oracle_lora_hi{nq//1000}k.ply")
            print(f"    -> exported {room}_{win.name}_oracle_lora_hi{nq//1000}k.ply", flush=True)

probe(CACHE/"office0/1085_1123_14f_41s", REPO/"outputs/replica/single_window_complete_train_office0", "TRAIN")
probe(CACHE/"office4/1028_1050_12f_28s", REPO/"outputs/replica/single_window_complete_val_office4",   "VAL")
print("DONE")
