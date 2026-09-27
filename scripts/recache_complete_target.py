"""Re-cache z* targets with the COMPLETE (amodal) frustum pool instead of the visible pool,
matching NOVA3R's src_complete training. Reuses each existing window's da3_tokens (symlink)
and meta; recomputes ONLY z* (30-seed online Hungarian) + pts_norm on the complete pool.
Preserves the per-window flip augmentation. Saves to a parallel *_complete cache."""
import sys, os, json, argparse, numpy as np, torch, trimesh
from pathlib import Path
REPO = Path("/usr/prakt/s0016/vc3r"); os.chdir(REPO)
for _p in ["nova3r_lib/third_party", "nova3r_lib", "da3/src", "scripts"]:
    sys.path.insert(0, _p)
os.environ.setdefault("NOVA3R_DIR", "nova3r_lib")
from demo_nova3r import load_model as load_nova3r_model
from cache_online_hungarian_zstar_windows import online_hungarian_zstar
from cache_consecutive_windows import world_to_first_camera
from vc3r.replica import project_world_points

DEV = torch.device("cuda")
NOVA_CKPT = REPO/"checkpoints/nova3r/scene_ae/checkpoint-last.pth"
N_SEEDS, VAR_FLOOR_Q, MESH_N = 30, 0.05, 2_000_000

ap = argparse.ArgumentParser()
ap.add_argument("--rooms", nargs="*", default=["office0","office1","office2","office3","room0","room1","room2","office4"])
ap.add_argument("--data-root", type=Path, default=Path("/usr/prakt/s0016/Replica"),
                help="dataset root (Replica layout) holding {room}_mesh.ply, cam_params.json")
ap.add_argument("--src", type=Path, default=REPO/"scripts/data/fc_nf16_span24_100_l13",
                help="source (visible) cache to reuse da3_tokens+meta from")
ap.add_argument("--dst", type=Path, default=REPO/"scripts/data/fc_nf16_span24_100_l13_complete",
                help="destination (complete) cache root")
ap.add_argument("--limit", type=int, default=None, help="cap windows per room (smoke test)")
args = ap.parse_args()
REPL, SRC, DST = args.data_root, args.src, args.dst

nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
[p.requires_grad_(False) for p in nova.parameters()]
norm_mode = ncfg.model.params.cfg.pts3d_head.params.get("norm_mode", "median_3")
cam = json.load(open(REPL/"cam_params.json"))["camera"]
H, W = int(cam["h"]), int(cam["w"])
Kmat = torch.tensor([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]])
print(f"[cfg] norm_mode={norm_mode} native HxW={H}x{W} n_seeds={N_SEEDS}", flush=True)

def frustum_crop(pts_world, c2w, K, hw):   # crop_visible WITHOUT the occlusion (depth) test
    proj = project_world_points(points_world=pts_world, world_to_camera=torch.linalg.inv(c2w),
                                intrinsics=K, image_hw=hw)
    return pts_world[proj["inside"]]

def astensor(x, dt=torch.float32):
    return x.float() if torch.is_tensor(x) else torch.tensor(np.array(x), dtype=dt)

for room in args.rooms:
    mesh = trimesh.load(str(REPL/f"{room}_mesh.ply"), force="mesh", process=False)
    mesh_pts = torch.from_numpy(trimesh.sample.sample_surface(mesh, MESH_N)[0].astype(np.float32))
    wins = sorted((SRC/room).glob("*/"))
    if args.limit:
        wins = wins[:args.limit]
    print(f"=== {room}: {len(wins)} windows ===", flush=True)
    for win in wins:
        win = Path(win); dst = DST/room/win.name
        if (dst/"z_star_online_mean.pt").exists():
            continue
        meta = torch.load(win/"meta.pt", weights_only=False)
        poses = meta["poses_c2w"]; first = astensor(meta["first_c2w"])
        pool_list = []
        for i in range(len(meta["frame_ids"])):
            c2w = astensor(poses[i])
            comp = frustum_crop(mesh_pts, c2w, Kmat, (H, W))
            pool_list.append(world_to_first_camera(comp, first))
        pool = torch.cat(pool_list, 0)                     # union over frames (no dedup, matches visible cache)
        pe = pool.clone()
        if bool(meta.get("flip", False)):
            pe[:, 0] = -pe[:, 0]                           # same L/R flip aug as the visible cache
        z = online_hungarian_zstar(nova, norm_mode, pe, DEV, n_seeds=N_SEEDS, var_floor_quantile=VAR_FLOOR_Q)
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(z["mean"],    dst/"z_star_online_mean.pt")
        torch.save(z["var"],     dst/"z_star_online_var.pt")
        torch.save(z["var_eff"], dst/"z_star_online_var_eff.pt")
        torch.save(z["aligned"], dst/"z_star_online_samples.pt")
        torch.save(z["pts_norm"],dst/"pts_norm.pt")
        m = dict(meta); m["norm_factor"] = z["norm_factor"]; m["var_floor"] = z["var_floor"]; m["complete_target"] = True
        torch.save(m, dst/"meta.pt")
        link = dst/"da3_tokens.pt"                          # REUSE da3 tokens (image-derived, unchanged)
        if not link.exists():
            os.symlink((win/"da3_tokens.pt").resolve(), link)
        print(f"  [{room}] {win.name} pool={len(pool):,} flip={bool(meta.get('flip',False))} nf={z['norm_factor']:.3f}", flush=True)
print("RECACHE DONE", flush=True)
