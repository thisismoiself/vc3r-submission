#!/usr/bin/env python3
"""
Compare k-means consensus vs Hungarian-aligned per-slot mean consensus.

For each test window, both methods encode 20 seeds. The decoded point cloud
from each is compared to the visible geometry (GT) via Chamfer distance.
Decode stochasticity (same z_star, two ODE seeds) gives the noise floor.

Usage:
  ! python experiments/test_hungarian_consensus.py
  ! python experiments/test_hungarian_consensus.py --starts 0 400 800 1200 1600 --room room0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from PIL import Image

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
SCRIPTS_SRC = REPO_ROOT / "scripts"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(SCRIPTS_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import trimesh
from demo_nova3r import load_model as load_nova3r_model
from nova3r.inference import normalize_input, amp_dtype_mapping
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from vc3r.replica import crop_visible_world_points

N_FRAMES   = 8
N_SEEDS    = 20
K          = 8192
N_CLUSTERS = 768
NUM_DECODE = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--starts", type=int, nargs="+", default=[0, 400, 800])
    p.add_argument("--room",   default="room0")
    p.add_argument("--replica-root", type=Path,
                   default=REPO_ROOT / "datasets" / "replica")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    return p.parse_args()


# ── geometry helpers ──────────────────────────────────────────────────────────

def world_to_first_camera(pts_world: torch.Tensor, first_c2w: torch.Tensor) -> torch.Tensor:
    w2c  = torch.linalg.inv(first_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


def sample_pts(pool: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if pool.shape[0] >= k:
        idx = torch.randperm(pool.shape[0], generator=gen)[:k]
    else:
        pad = torch.randint(pool.shape[0], (k - pool.shape[0],), generator=gen)
        idx = torch.cat([torch.arange(pool.shape[0]), pad])
    return pool[idx]


@torch.no_grad()
def encode_once(model, norm_mode: str, pts: torch.Tensor, device: torch.device):
    pts_dev = pts.unsqueeze(0).to(device).float()
    valid   = torch.ones(pts_dev.shape[:2], dtype=torch.bool, device=device)
    pts_norm, _ = normalize_input(pts_dev, valid, pts_dev, valid, mode=norm_mode)
    tokens  = model._encode(pointmaps=pts_norm, test=True)["tokens"].float().cpu()
    nf      = float(pts_dev.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))
    return tokens[0], pts_norm.cpu(), nf


def encode_all_seeds(model, norm_mode, pool, device):
    tokens_list, pts_norm_0, nf_0 = [], None, None
    for seed in range(N_SEEDS):
        z, pnorm, nf = encode_once(model, norm_mode, sample_pts(pool, K, seed), device)
        tokens_list.append(z)
        if seed == 0:
            pts_norm_0, nf_0 = pnorm, nf
    return tokens_list, pts_norm_0, nf_0


# ── consensus methods ─────────────────────────────────────────────────────────

def kmeans_consensus(tokens_list: list[torch.Tensor]) -> torch.Tensor:
    flat = torch.stack(tokens_list).reshape(N_SEEDS * N_CLUSTERS, -1).numpy()
    km   = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=5, max_iter=300, verbose=0)
    km.fit(flat)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32)).unsqueeze(0)


def hungarian_mean_consensus(tokens_list: list[torch.Tensor]) -> torch.Tensor:
    """Hungarian-align seeds 1..N-1 to seed-0, then per-slot mean."""
    ref     = tokens_list[0]   # (768, 128)
    aligned = [ref]
    for z in tokens_list[1:]:
        cost = torch.cdist(ref.unsqueeze(0), z.unsqueeze(0))[0].numpy()  # (768, 768)
        _, col_ind = linear_sum_assignment(cost)
        aligned.append(z[col_ind])
    return torch.stack(aligned).mean(0).unsqueeze(0)  # (1, 768, 128)


# ── decode ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens: torch.Tensor,
                  pts_norm: torch.Tensor, device: torch.device, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, NUM_DECODE, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    use_amp = device.type != "cpu"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def norm_to_world(pts_norm_np: np.ndarray, norm_factor: float, c2w: np.ndarray) -> np.ndarray:
    pts_cam = pts_norm_np / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


# ── Chamfer distance ──────────────────────────────────────────────────────────

def chamfer(a: np.ndarray, b: np.ndarray, subsample: int = 10_000) -> float:
    rng = np.random.default_rng(0)
    if len(a) > subsample:
        a = a[rng.choice(len(a), subsample, replace=False)]
    if len(b) > subsample:
        b = b[rng.choice(len(b), subsample, replace=False)]
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    d   = torch.cdist(a_t, b_t)
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    device      = torch.device(args.device)
    replica_root = args.replica_root
    room_dir    = replica_root / args.room
    results_dir = room_dir / "results"
    mesh_ply    = replica_root / f"{args.room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    H_nat, W_nat  = int(cam["h"]), int(cam["w"])
    depth_scale   = float(cam["scale"])
    K_native      = torch.tensor([
        [cam["fx"], 0., cam["cx"]],
        [0., cam["fy"], cam["cy"]],
        [0., 0., 1.]], dtype=torch.float32)

    poses_all   = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]

    print("[test] Loading mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    print("[test] Loading NOVA3R …")
    cfg_path  = OVERFIT_SRC / "config.yaml"
    nova_cfg_omg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(nova_cfg_omg, False)
    ckpt      = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    header = f"{'start':>6}  {'kmeans':>10}  {'hungarian':>10}  {'noise_floor':>11}  {'winner':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    results = []
    for start in args.starts:
        frame_ids = [all_frame_ids[start + i] for i in range(N_FRAMES)]
        poses_c2w = torch.from_numpy(
            np.stack([poses_all[fid] for fid in frame_ids])).float()
        first_c2w = poses_c2w[0]
        c2w_np    = first_c2w.numpy()

        # Build visible geometry pool in first-camera frame
        pool_list = []
        for i, fid in enumerate(frame_ids):
            d = Image.open(results_dir / f"depth{fid:06d}.png")
            depth = torch.from_numpy(np.asarray(d, dtype=np.float32) / depth_scale)
            vis = crop_visible_world_points(
                points_world=mesh_pts,
                camera_to_world=poses_c2w[i],
                intrinsics=K_native,
                depth=depth,
                depth_tolerance=args.depth_tolerance,
            )
            pool_list.append(world_to_first_camera(vis["points_world"], first_c2w))
        pool = torch.cat(pool_list, dim=0)

        # GT in world space (downsampled pool, seed-0 subsample)
        gt_cam = sample_pts(pool, 50_000, seed=99).numpy()
        ones   = np.ones((len(gt_cam), 1), dtype=np.float32)
        nf_gt  = float(torch.from_numpy(gt_cam).norm(dim=-1).median().clamp(0.01, 100.0))
        gt_world = (first_c2w.numpy() @ np.hstack([gt_cam, ones]).T).T[:, :3]

        print(f"  start={start:04d}  pool={len(pool):,}  encoding {N_SEEDS} seeds …", flush=True)
        tokens_list, pts_norm_0, nf_0 = encode_all_seeds(nova_model, norm_mode, pool, device)

        print(f"    k-means consensus …", flush=True)
        z_km  = kmeans_consensus(tokens_list)

        print(f"    Hungarian-mean consensus …", flush=True)
        z_hun = hungarian_mean_consensus(tokens_list)

        print(f"    decoding …", flush=True)
        dec_km_a  = norm_to_world(decode_tokens(nova_model, nova_cfg, z_km,  pts_norm_0, device, seed=0), nf_0, c2w_np)
        dec_km_b  = norm_to_world(decode_tokens(nova_model, nova_cfg, z_km,  pts_norm_0, device, seed=1), nf_0, c2w_np)
        dec_hun_a = norm_to_world(decode_tokens(nova_model, nova_cfg, z_hun, pts_norm_0, device, seed=0), nf_0, c2w_np)

        cd_km    = chamfer(dec_km_a, gt_world)
        cd_hun   = chamfer(dec_hun_a, gt_world)
        cd_noise = chamfer(dec_km_a, dec_km_b)
        winner   = "hungarian" if cd_hun < cd_km else "kmeans"

        results.append(dict(start=start, cd_km=cd_km, cd_hun=cd_hun,
                            cd_noise=cd_noise, winner=winner))
        print(f"  {start:6d}  {cd_km:10.4f}  {cd_hun:10.4f}  {cd_noise:11.4f}  {winner:>8}")

    print()
    print(f"{'mean':>6}  "
          f"{np.mean([r['cd_km']  for r in results]):10.4f}  "
          f"{np.mean([r['cd_hun'] for r in results]):10.4f}  "
          f"{np.mean([r['cd_noise'] for r in results]):11.4f}")


if __name__ == "__main__":
    main()
