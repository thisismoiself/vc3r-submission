"""Export flow-matching decode intermediates for an ACADEMIC figure.

The NOVA3R decoder IS a flow-matching model: it starts from a uniform-noise point
cloud at t=0 and integrates an ODE to t=1 to produce the scene. We tap the solver's
intermediate states (same noise, one trajectory per point) and dump each as a
world-frame .ply, plus render a matplotlib panel figure (noise -> ... -> scene).

Tokens = ORACLE tokens (encode of the real complete-frustum geometry), so the final
cloud is the crisp AE ceiling — the cleanest thing to show for "what FM produces".

Point IDENTITY is preserved across ODE steps, so every panel is coloured by the SAME
per-point value (final-cloud height): you literally watch coloured points migrate from
a noise blob into the room.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np, torch, trimesh

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r"); NOVA = REPO / "nova3r_lib"
for p in [str(NOVA / "third_party"), str(NOVA), str(REPO / "da3" / "src"),
          str(REPO / "scripts"), str(REPO / "experiments" / "overfit_8frames")]:
    if p not in sys.path: sys.path.insert(0, p)

from demo_nova3r import load_model as load_nova3r_model
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from nova3r.inference import normalize_input
from vc3r.replica import project_world_points, crop_visible_world_points
from multi_scene_train import world_to_first_camera
from omegaconf import OmegaConf

NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_SAMPLE = 8192

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="/usr/prakt/s0016/Replica")
ap.add_argument("--room", default="office4")
ap.add_argument("--frames", default="1028,1031,1034,1037,1040,1043,1046,1049")
ap.add_argument("--num-queries", type=int, default=30000)
ap.add_argument("--n-steps", type=int, default=50, help="ODE integration steps (euler/midpoint)")
ap.add_argument("--snap-t", default="0.0,0.2,0.4,0.6,0.8,1.0", help="t values to export")
ap.add_argument("--method", default="midpoint")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--elev", type=float, default=22.0)
ap.add_argument("--azim", type=float, default=-70.0)
ap.add_argument("--render-pts", type=int, default=15000, help="subsample for the figure")
ap.add_argument("--out", default="outputs/replica/fm_intermediates")
args = ap.parse_args()

out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
fids = [int(x) for x in args.frames.split(",")]
snap_t = [float(x) for x in args.snap_t.split(",")]

print("[load] NOVA3R"); nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV))
nova.eval(); [pp.requires_grad_(False) for pp in nova.parameters()]
OmegaConf.set_struct(ncfg, False)
norm_mode = ncfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

import json
cam = json.load(open(Path(args.data_root) / "cam_params.json"))["camera"]
H_nat, W_nat = int(cam["h"]), int(cam["w"])
K_native = torch.tensor([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1.]], dtype=torch.float32)

room_dir = Path(args.data_root) / args.room
poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
mesh = trimesh.load(str(Path(args.data_root) / f"{args.room}_mesh.ply"), force="mesh", process=False)
mesh_pts = torch.from_numpy(trimesh.sample.sample_surface(mesh, 2_000_000)[0].astype(np.float32))

poses_c2w = torch.from_numpy(np.stack([poses_all[f] for f in fids])).float()
first_c2w = poses_c2w[0]

# complete/amodal frustum pool in first-camera frame
pool_list = []
for i, fid in enumerate(fids):
    proj = project_world_points(points_world=mesh_pts,
        world_to_camera=torch.linalg.inv(poses_c2w[i]), intrinsics=K_native, image_hw=(H_nat, W_nat))
    pool_list.append(world_to_first_camera(mesh_pts[proj["inside"]], first_c2w))
pool = torch.cat(pool_list, 0)
print(f"[pool] complete frustum pool {pool.shape[0]:,} pts")

gen = torch.Generator().manual_seed(0)
idx = torch.randperm(pool.shape[0], generator=gen)[:K_SAMPLE]
pts = pool[idx].unsqueeze(0).to(DEV).float()
valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=DEV)
with torch.no_grad():
    pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
    z_enc = nova._encode(pointmaps=pts_norm, test=True)["tokens"].float()   # ORACLE tokens
nf = float(pts.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))
R, t = first_c2w[:3, :3].numpy(), first_c2w[:3, 3].numpy()
to_world = lambda pn: (R @ (pn / 3.0 * nf).T).T + t

# ---- run the FM ODE, capturing ALL intermediate states ----
torch.manual_seed(args.seed)
x_init = torch.rand(1, args.num_queries, 3, device=DEV) * 2 - 1
solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
time_grid = torch.linspace(0, 1, args.n_steps + 1).to(DEV)
with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
    sol = solver.sample(time_grid=time_grid, x_init=x_init, method=args.method,
                        step_size=1.0 / args.n_steps, return_intermediates=True,
                        images=torch.zeros(1, 1, 3, 1, 1, device=DEV), token_mask=None,
                        encoder_data={"tokens": z_enc}, pointmaps=pts_norm)
sol = torch.stack(sol) if isinstance(sol, (list, tuple)) else sol   # [T+1,1,Q,3]
sol = sol[:, 0].cpu().float().numpy()                               # [T+1,Q,3]
print(f"[fm] captured {sol.shape[0]} states, {sol.shape[1]:,} points each")

# snapshot indices
snap_idx = [int(round(tt * args.n_steps)) for tt in snap_t]
clouds_world = [to_world(sol[k]) for k in snap_idx]

# colour by FINAL-cloud height (same per-point colour across all panels)
final = clouds_world[-1]
h = final[:, 2]
hn = (h - np.percentile(h, 2)) / (np.percentile(h, 98) - np.percentile(h, 2) + 1e-9)
hn = np.clip(hn, 0, 1)

# ---- export plys ----
for tt, cw in zip(snap_t, clouds_world):
    import matplotlib.cm as cm
    col = (cm.viridis(hn)[:, :3] * 255).astype(np.uint8)
    trimesh.PointCloud(cw, colors=col).export(out / f"fm_t{tt:.2f}.ply")
np.savez(out / "fm_states.npz", clouds=np.stack(clouds_world), t=np.array(snap_t),
         hn=hn, input_pool=to_world(pts_norm[0].cpu().numpy()))
print(f"[out] {len(snap_t)} plys -> {out}")

# ---- academic matplotlib figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
ri = rng.choice(final.shape[0], min(args.render_pts, final.shape[0]), replace=False)
n = len(snap_t)
fig = plt.figure(figsize=(3.0 * n, 3.4))
# shared axis limits from the final cloud (a bit padded)
lim = np.stack([final[ri].min(0), final[ri].max(0)])
ctr = lim.mean(0); rad = (lim[1] - lim[0]).max() * 0.62
for j, (tt, cw) in enumerate(zip(snap_t, clouds_world)):
    ax = fig.add_subplot(1, n, j + 1, projection="3d")
    ax.scatter(cw[ri, 0], cw[ri, 1], cw[ri, 2], c=hn[ri], cmap="viridis",
               s=1.2, alpha=0.85, linewidths=0, vmin=0, vmax=1)
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    label = (r"$t=0$  (noise)" if tt == 0 else (r"$t=1$  (scene)" if tt == 1 else fr"$t={tt:g}$"))
    ax.set_title(label, fontsize=12, pad=-2)
fig.suptitle("Flow-matching decode: uniform noise " + r"$\longrightarrow$" +
             " scene (NOVA3R, office4)", y=0.99, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(out / f"fm_flow_panel.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {out}/fm_flow_panel.png / .pdf")
