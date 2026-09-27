#!/usr/bin/env python3
"""CEILING PROBE (no training): does routing DA3-predicted GEOMETRY through the FROZEN
NOVA3R encoder beat the image->latent adapter?

    GT oracle :  decode(encode(normalize(GT complete-frustum points)))     -> ceiling
    DA3-encode:  decode(encode(normalize(DA3 depth points, GT-free)))       -> the new path
    (adapter image->z* baseline is ~0.31 FURN F@2 from prior runs)

Both point sources are put in the SAME first-camera metric frame and normalized identically
(normalize_input, median_3), so the ONLY difference is GT-mesh points vs DA3-depth points.
If DA3-encode lands well above 0.31, the frozen encoder + DA3 geometry is a real path and the
remaining GT-free->complete gap is a uni-modal latent-completion problem.
"""
import os, sys
from pathlib import Path
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r"); NOVA = REPO / "nova3r_lib"
for p in [str(NOVA / "third_party"), str(NOVA), str(REPO / "da3" / "src"),
          str(REPO / "scripts"), str(REPO / "experiments" / "overfit_8frames")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from omegaconf import OmegaConf
from PIL import Image
from demo_nova3r import load_model as load_nova3r_model
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from nova3r.inference import normalize_input
from vc3r.replica import project_world_points
from multi_scene_train import world_to_first_camera
from stitch_office4 import load_da3_depth_model

DEV = torch.device("cuda")
NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
ROOM_DIR = Path("/usr/prakt/s0016/Replica/office4")
IMG_H, IMG_W = 392, 518
K_SAMPLE = 8192
N_WINDOWS = 6
STRIDE, NF = 10, 8


@torch.no_grad()
def decode(nova, ncfg, tokens, pts_norm, nq, seed=42):
    torch.manual_seed(seed); xi = torch.rand(1, nq, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=xi, method="midpoint", step_size=step,
                            return_intermediates=False, images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": tokens.to(DEV)}, pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def sample_pad(pts, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    if pts.shape[0] >= k:
        idx = torch.randperm(pts.shape[0], generator=g)[:k]
    else:
        idx = torch.cat([torch.arange(pts.shape[0]), torch.randint(pts.shape[0], (k - pts.shape[0],), generator=g)])
    return pts[idx]


def encode_decode(nova, ncfg, norm_mode, pts_fc, nq):
    """pts_fc: [K,3] first-camera metric points -> normalize_input -> encode -> decode."""
    pts = sample_pad(pts_fc, K_SAMPLE).unsqueeze(0).to(DEV).float()
    valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=DEV)
    with torch.no_grad():
        pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
        z = nova._encode(pointmaps=pts_norm, test=True)["tokens"].float()
    nf = float(pts.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))
    cloud_n = decode(nova, ncfg, z.cpu(), pts_norm.cpu(), nq)
    return cloud_n, pts_norm, nf


def da3_first_cam_points(model, results, fids, poses_c2w, first_c2w, K_native):
    """DA3 depth -> world (GT poses) -> first-camera metric frame (GT-free geometry)."""
    from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
    paths = [str(results / f"frame{f:06d}.jpg") for f in fids]
    c2w = poses_c2w.numpy().astype(np.float32); w2c = np.linalg.inv(c2w)
    Kin = np.tile(K_native.numpy()[None], (len(fids), 1, 1))
    with torch.no_grad():
        p = model.inference(image=paths, extrinsics=w2c, intrinsics=Kin,
                            align_to_input_ext_scale=True, process_res=504)
    depth = np.asarray(p.depth); conf = None if p.conf is None else np.asarray(p.conf)
    Kp = np.asarray(p.intrinsics); H, W = depth.shape[-2:]
    cthr = np.percentile(conf, 40.0) if conf is not None else 0.0
    da3w, _ = _depths_to_world_points_with_colors(
        depth, Kp, w2c, np.zeros((len(fids), H, W, 3), np.uint8), conf, cthr)
    da3w = da3w[np.isfinite(da3w).all(1)]
    return world_to_first_camera(torch.from_numpy(da3w).float(), first_c2w)


def main():
    import json
    nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
    OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
    norm_mode = ncfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    da3 = load_da3_depth_model()

    cam = json.load(open("/usr/prakt/s0016/Replica/cam_params.json"))["camera"]
    H_nat, W_nat = int(cam["h"]), int(cam["w"])
    K_native = torch.tensor([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=torch.float32)
    results = ROOM_DIR / "results"
    poses_all = np.loadtxt(ROOM_DIR / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    n_total = len(list(sorted(results.glob("frame*.jpg"))))

    mesh = trimesh.load(str(Path("/usr/prakt/s0016/Replica/office4_mesh.ply")), force="mesh", process=False)
    mesh_pts = torch.from_numpy(trimesh.sample.sample_surface(mesh, 2_000_000)[0].astype(np.float32))
    gt_pts = np.asarray(trimesh.load(str(REPO / "outputs/replica/gt_pointclouds/office4/office4_gt_2m.ply"),
                                     process=False).vertices, np.float32)
    z = gt_pts[:, 2]; floor = float(np.percentile(z, 1))
    xy_lo = np.percentile(gt_pts[:, :2], 1, axis=0); xy_hi = np.percentile(gt_pts[:, :2], 99, axis=0)
    def fmask(p, margin=0.4, zlo=0.15, zhi=1.1):
        zz = p[:, 2]
        return ((zz > floor + zlo) & (zz < floor + zhi) & (p[:, 0] > xy_lo[0] + margin)
                & (p[:, 0] < xy_hi[0] - margin) & (p[:, 1] > xy_lo[1] + margin) & (p[:, 1] < xy_hi[1] - margin))
    def furn_f2(cloud_w, gt_win):
        """Compare a window's decode to THAT window's own complete GT (fair per-window recall)."""
        c = cloud_w[fmask(cloud_w)]; g = gt_win[fmask(gt_win)]
        if len(c) == 0 or len(g) == 0: return None
        d_pg = cKDTree(g).query(c)[0]; d_gp = cKDTree(c).query(g)[0]
        prec = (d_pg < 0.02).mean(); rec = (d_gp < 0.02).mean()
        return 2 * prec * rec / (prec + rec + 1e-9), prec, rec

    span = (NF - 1) * STRIDE; windows = []; base = 0
    while base + span < n_total and len(windows) < N_WINDOWS:
        windows.append([base + j * STRIDE for j in range(NF)]); base += span + STRIDE

    agg = {"ORACLE(GT)": [], "DA3-encode": []}
    for wi, fids in enumerate(windows):
        poses_c2w = torch.from_numpy(np.stack([poses_all[f] for f in fids])).float(); first_c2w = poses_c2w[0]
        # GT complete-frustum pool in first-cam frame
        pool = []
        for i in range(NF):
            proj = project_world_points(points_world=mesh_pts, world_to_camera=torch.linalg.inv(poses_c2w[i]),
                                        intrinsics=K_native, image_hw=(H_nat, W_nat))
            pool.append(world_to_first_camera(mesh_pts[proj["inside"]], first_c2w))
        gt_fc = torch.cat(pool, 0)
        da3_fc = da3_first_cam_points(da3, results, fids, poses_c2w, first_c2w, K_native)

        R, t = first_c2w[:3, :3].numpy(), first_c2w[:3, 3].numpy()
        gt_win_w = (R @ gt_fc.numpy().T).T + t            # this window's COMPLETE GT in world
        msg = [f"  win {wi} frames {fids[0]}-{fids[-1]}"]
        for tag, pts_fc in [("ORACLE(GT)", gt_fc), ("DA3-encode", da3_fc)]:
            cloud_n, _, nf = encode_decode(nova, ncfg, norm_mode, pts_fc, 50000)
            cloud_w = (R @ (cloud_n / 3.0 * nf).T).T + t
            res = furn_f2(cloud_w, gt_win_w)
            if res is not None:
                agg[tag].append(res); msg.append(f"{tag} F@2={res[0]:.3f}")
        print("  ".join(msg), flush=True)

    print(f"\n=== office4 {len(windows)} windows, per-window FURN F@2 (world 2cm) ===")
    print(f"  (image->z* adapter baseline ~0.31 for reference)")
    for tag in ["ORACLE(GT)", "DA3-encode"]:
        a = np.array(agg[tag])
        print(f"  {tag:12s}  F@2={a[:,0].mean():.4f}  prec={a[:,1].mean():.3f}  rec={a[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
