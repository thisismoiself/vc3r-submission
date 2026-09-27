"""DECISIVE TEST: does encoding the COMPLETE frustum pool (NOVA3R's src_complete design)
make the oracle fill occlusions, vs the current visible-only pool?
One office4 window (non-flipped). Compare decode(encode(visible)) vs decode(encode(complete))
against the COMPLETE frustum GT. Export all clouds."""
import sys, os, json, numpy as np, torch, trimesh
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
REPO = Path("/usr/prakt/s0016/vc3r"); os.chdir(REPO)
for _p in ["nova3r_lib/third_party", "nova3r_lib", "da3/src", "scripts"]:
    sys.path.insert(0, _p)
os.environ.setdefault("NOVA3R_DIR", "nova3r_lib")
from cache_consecutive_windows import encode_once, crop_visible_world_points, world_to_first_camera, load_nova3r_model
from vc3r.replica import project_world_points
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from omegaconf import OmegaConf
DEV = torch.device("cuda")
NOVA_CKPT = REPO/"checkpoints/nova3r/scene_ae/checkpoint-last.pth"
REPL = Path("/usr/prakt/s0016/Replica")
WIN = REPO/"scripts/data/fc_nf16_span24_100_l13/office4/1028_1050_12f_28s"   # non-flipped
NQ, SEED, CAP = 50000, 42, 250_000

nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
[p.requires_grad_(False) for p in nova.parameters()]
OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"

def frustum_crop(points_world, c2w, K, hw):   # like crop_visible but NO occlusion cull
    proj = project_world_points(points_world=points_world, world_to_camera=torch.linalg.inv(c2w),
                                intrinsics=K, image_hw=hw)
    return points_world[proj["inside"]]

@torch.no_grad()
def decode(tokens, pts_norm):
    torch.manual_seed(SEED); xi = torch.rand(1, NQ, 3, device=DEV)*2-1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1//step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=xi, method="midpoint", step_size=step,
            return_intermediates=False, images=torch.zeros(1,1,3,1,1,device=DEV),
            token_mask=None, encoder_data={"tokens": tokens.unsqueeze(0).to(DEV)}, pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()

meta = torch.load(WIN/"meta.pt", weights_only=False)
fids = meta["frame_ids"]
first = torch.tensor(np.array(meta["first_c2w"]), dtype=torch.float32)
cam = json.load(open(REPL/"cam_params.json"))["camera"]; ds = float(cam["scale"])
K = torch.tensor([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]])
poses = np.loadtxt(REPL/f"{meta['room']}/traj.txt", np.float32).reshape(-1,4,4)
mesh = trimesh.load(str(REPL/f"{meta['room']}_mesh.ply"), force="mesh", process=False)
mp = torch.from_numpy(trimesh.sample.sample_surface(mesh, 2_000_000)[0].astype(np.float32))
def ld(fid): return torch.from_numpy(np.asarray(Image.open(REPL/f"{meta['room']}/results"/f"depth{fid:06d}.png"), np.float32)/ds)
H, W = int(cam["h"]), int(cam["w"])

vis_list, comp_list = [], []
for fid in fids:
    c2w = torch.from_numpy(poses[fid])
    vis = crop_visible_world_points(points_world=mp, camera_to_world=c2w, intrinsics=K, depth=ld(fid), depth_tolerance=0.05)["points_world"]
    comp = frustum_crop(mp, c2w, K, (H, W))
    vis_list.append(world_to_first_camera(vis, first))
    comp_list.append(world_to_first_camera(comp, first))
vis_pool = torch.cat(vis_list, 0)
comp_pool = torch.unique(torch.cat(comp_list, 0), dim=0)   # dedup overlapping frustums
def cap(p):
    return p if len(p) <= CAP else p[torch.randperm(len(p))[:CAP]]
vis_pool, comp_pool = cap(vis_pool), cap(comp_pool)
print(f"[pools] visible={len(vis_pool):,}  complete(frustum)={len(comp_pool):,}", flush=True)

zv, pnv, nfv = encode_once(nova, "median_3", vis_pool, DEV)
zc, pnc, nfc = encode_once(nova, "median_3", comp_pool, DEV)
dec_v = decode(zv, pnv) * (nfv/3.0)
dec_c = decode(zc, pnc) * (nfc/3.0)
gt_complete = comp_pool.numpy()          # complete frustum GT (cam1 metric)
gt_visible  = vis_pool.numpy()

def comp_metric(dec, gt):   # completeness = GT->pred mean (how well the surface is covered)
    return cKDTree(dec).query(gt, k=1)[0].mean()
cv_full = comp_metric(dec_v, gt_complete); cc_full = comp_metric(dec_c, gt_complete)
print(f"\n=== completeness to the COMPLETE frustum GT (lower=fewer holes) ===")
print(f"  oracle from VISIBLE pool (current) : {cv_full*100:.2f} cm")
print(f"  oracle from COMPLETE pool (fix)    : {cc_full*100:.2f} cm")
print(f"  visible GT   -> complete GT (ref)  : {comp_metric(gt_visible, gt_complete)*100:.2f} cm  (how holey the visible input itself is)")

out = REPO/"outputs/replica/complete_target_test_office4_w1028"
out.mkdir(parents=True, exist_ok=True)
trimesh.PointCloud(dec_v).export(out/"oracle_from_visible.ply")
trimesh.PointCloud(dec_c).export(out/"oracle_from_complete.ply")
trimesh.PointCloud(gt_visible).export(out/"gt_visible.ply")
trimesh.PointCloud(gt_complete).export(out/"gt_complete.ply")
print(f"\nexported -> {out}")
print("DONE")
