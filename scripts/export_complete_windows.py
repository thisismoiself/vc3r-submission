"""Single-window pred/oracle/GT clouds for the COMPLETE-target run, train + val window.
pred = complete-target adapter, oracle = decode(complete z*), GT = COMPLETE frustum crop
(amodal, so the completion is visible). Dense 200k, chunked decode. Non-flipped windows."""
import sys, os, json, numpy as np, torch, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
REPO = Path("/usr/prakt/s0016/vc3r"); os.chdir(REPO)
for _p in ["nova3r_lib/third_party", "nova3r_lib", "da3/src", "scripts"]:
    sys.path.insert(0, _p)
os.environ.setdefault("NOVA3R_DIR", "nova3r_lib")
from demo_nova3r import load_model as load_nova3r_model
from vc3r.alignment import DA3ToNOVA3RAlignment
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from cache_consecutive_windows import world_to_first_camera
from vc3r.replica import project_world_points
from omegaconf import OmegaConf
DEV = torch.device("cuda")
NOVA_CKPT = REPO/"checkpoints/nova3r/scene_ae/checkpoint-last.pth"
CKPT = REPO/"outputs/consecutive_windows/fullrec_nf16_vel03_complete_best.pt"
CACHE = REPO/"scripts/data/fc_nf16_span24_100_l13_complete"
REPL = Path("/usr/prakt/s0016/Replica")
NQ_TOTAL, CHUNK, SEED, MESH_N = 200_000, 50_000, 42, 4_000_000

nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
[p.requires_grad_(False) for p in nova.parameters()]
OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
def acfg(sd):
    tt, hd = sd["target_queries"].shape
    return dict(source_dim=sd["source_proj.weight"].shape[1], hidden_dim=hd, target_tokens=tt,
                target_dim=sd["out_proj.weight"].shape[0],
                depth=sum(1 for k in sd if k.endswith(".query_norm.weight")), num_heads=hd//64)
adapter = DA3ToNOVA3RAlignment(**acfg(sd), drop=0.0).to(DEV); adapter.load_state_dict(sd); adapter.eval()
cam = json.load(open(REPL/"cam_params.json"))["camera"]
Hn, Wn = int(cam["h"]), int(cam["w"])
Kmat = torch.tensor([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]])

@torch.no_grad()
def decode_dense(tokens, pts_norm):
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1//step)).to(DEV)
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova)); outs = []
    for ci, s in enumerate(range(0, NQ_TOTAL, CHUNK)):
        n = min(CHUNK, NQ_TOTAL - s); torch.manual_seed(SEED + ci)
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

def export(win, out_dir, tag):
    win, out_dir = Path(win), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = torch.load(win/"meta.pt", weights_only=False)
    da3 = torch.load(win/"da3_tokens.pt", weights_only=False).float()
    zmean = torch.load(win/"z_star_online_mean.pt", weights_only=False).float()
    pts_norm = torch.load(win/"pts_norm.pt", weights_only=False).float()
    nf = float(meta["norm_factor"]); room = meta["room"]; fids = meta["frame_ids"]
    poses = meta["poses_c2w"]; first = astensor(meta["first_c2w"])
    with torch.no_grad(): zpred = adapter(da3.to(DEV)).cpu()
    scale = nf/3.0
    pred = decode_dense(zpred, pts_norm) * scale
    orac = decode_dense(zmean, pts_norm) * scale
    mesh = trimesh.load(str(REPL/f"{room}_mesh.ply"), force="mesh", process=False)
    mp = torch.from_numpy(trimesh.sample.sample_surface(mesh, MESH_N)[0].astype(np.float32))
    gt = torch.cat([world_to_first_camera(frustum(mp, astensor(poses[i]), (Hn, Wn)), first)
                    for i in range(len(fids))], 0)
    gt = torch.unique(torch.round(gt*1e4)/1e4, dim=0).numpy()
    trimesh.PointCloud(pred).export(out_dir/f"{room}_{win.name}_pred.ply")
    trimesh.PointCloud(orac).export(out_dir/f"{room}_{win.name}_oracle.ply")
    trimesh.PointCloud(gt).export(out_dir/f"{room}_{win.name}_gt_complete.ply")
    ep = cKDTree(gt).query(pred, k=1)[0].mean(); cp = cKDTree(pred).query(gt, k=1)[0].mean()
    print(f"[{tag} {room}] {win.name} acc(pred->GT)={ep*100:.2f}cm comp(GT->pred)={cp*100:.2f}cm | "
          f"200k pred/oracle, GT {len(gt):,} -> {out_dir}", flush=True)

export(CACHE/"office0/1085_1123_14f_41s", REPO/"outputs/replica/single_window_complete_train_office0", "TRAIN")
export(CACHE/"office4/1028_1050_12f_28s", REPO/"outputs/replica/single_window_complete_val_office4", "VAL")
print("DONE")
