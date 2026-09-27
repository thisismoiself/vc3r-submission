#!/usr/bin/env python3
"""
Measure z_star encode-variance vs decode noise floor.

For each window, encode 20 seeds and split into two halves (seeds 0–9 and 10–19).
Build a k-means consensus from each half, decode both with the same ODE seed,
and measure Chamfer between the two decoded point clouds.

This answers: is z_star variation across re-encodings large compared to ODE
decode stochasticity? If yes, the z_star distribution is meaningfully multi-modal
and flow matching would add value. If no, deterministic regression is sufficient.

Usage:
  ! python experiments/test_zstar_variance.py
  ! python experiments/test_zstar_variance.py --starts 0 400 800 1200 1600 --room room0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
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
N_SEEDS    = 20          # total seeds; split into two halves of 10
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
def encode_once(model, norm_mode, pts, device):
    pts_dev = pts.unsqueeze(0).to(device).float()
    valid   = torch.ones(pts_dev.shape[:2], dtype=torch.bool, device=device)
    pts_norm, _ = normalize_input(pts_dev, valid, pts_dev, valid, mode=norm_mode)
    tokens  = model._encode(pointmaps=pts_norm, test=True)["tokens"].float().cpu()
    nf      = float(pts_dev.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))
    return tokens[0], pts_norm.cpu(), nf


def kmeans_consensus(tokens_list: list[torch.Tensor]) -> torch.Tensor:
    n = len(tokens_list)
    flat = torch.stack(tokens_list).reshape(n * N_CLUSTERS, -1).numpy()
    km   = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=5, max_iter=300, verbose=0)
    km.fit(flat)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32)).unsqueeze(0)


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens, pts_norm, device, seed: int) -> np.ndarray:
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


def norm_to_world(pts_norm_np, norm_factor, c2w):
    pts_cam = pts_norm_np / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


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


def main() -> None:
    args         = parse_args()
    device       = torch.device(args.device)
    replica_root = args.replica_root
    room_dir     = replica_root / args.room
    results_dir  = room_dir / "results"
    mesh_ply     = replica_root / f"{args.room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    H_nat, W_nat = int(cam["h"]), int(cam["w"])
    depth_scale  = float(cam["scale"])
    K_native     = torch.tensor([
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
    cfg_path     = OVERFIT_SRC / "config.yaml"
    nova_cfg_omg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(nova_cfg_omg, False)
    ckpt         = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    half = N_SEEDS // 2   # 10 seeds per half

    print(f"\nz_star variance test — {args.room}, {N_SEEDS} seeds split into two halves of {half}")
    print(f"  encode_variance  = Chamfer(decode(z_A), decode(z_B))  — different seed halves, same ODE seed")
    print(f"  decode_noise     = Chamfer(decode(z_full, ode=0), decode(z_full, ode=1))  — same z, different ODE")
    header = f"{'start':>6}  {'encode_var':>12}  {'decode_noise':>12}  {'ratio':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    results = []
    for start in args.starts:
        frame_ids = [all_frame_ids[start + i] for i in range(N_FRAMES)]
        poses_c2w = torch.from_numpy(
            np.stack([poses_all[fid] for fid in frame_ids])).float()
        first_c2w = poses_c2w[0]
        c2w_np    = first_c2w.numpy()

        pool_list = []
        for i, fid in enumerate(frame_ids):
            d     = Image.open(results_dir / f"depth{fid:06d}.png")
            depth = torch.from_numpy(np.asarray(d, dtype=np.float32) / depth_scale)
            vis   = crop_visible_world_points(
                points_world=mesh_pts,
                camera_to_world=poses_c2w[i],
                intrinsics=K_native,
                depth=depth,
                depth_tolerance=args.depth_tolerance,
            )
            pool_list.append(world_to_first_camera(vis["points_world"], first_c2w))
        pool = torch.cat(pool_list, dim=0)

        print(f"  start={start:04d}  pool={len(pool):,}  encoding {N_SEEDS} seeds …", flush=True)
        tokens_list, pts_norm_0, nf_0 = [], None, None
        for seed in range(N_SEEDS):
            z, pnorm, nf = encode_once(nova_model, norm_mode,
                                       sample_pts(pool, K, seed), device)
            tokens_list.append(z)
            if seed == 0:
                pts_norm_0, nf_0 = pnorm, nf

        print(f"    k-means on each half …", flush=True)
        z_A    = kmeans_consensus(tokens_list[:half])       # seeds 0–9
        z_B    = kmeans_consensus(tokens_list[half:])       # seeds 10–19
        z_full = kmeans_consensus(tokens_list)              # all 20 seeds

        print(f"    decoding …", flush=True)
        dec_A      = norm_to_world(decode_tokens(nova_model, nova_cfg, z_A,    pts_norm_0, device, seed=0), nf_0, c2w_np)
        dec_B      = norm_to_world(decode_tokens(nova_model, nova_cfg, z_B,    pts_norm_0, device, seed=0), nf_0, c2w_np)
        dec_full_0 = norm_to_world(decode_tokens(nova_model, nova_cfg, z_full, pts_norm_0, device, seed=0), nf_0, c2w_np)
        dec_full_1 = norm_to_world(decode_tokens(nova_model, nova_cfg, z_full, pts_norm_0, device, seed=1), nf_0, c2w_np)

        cd_encode = chamfer(dec_A, dec_B)
        cd_decode = chamfer(dec_full_0, dec_full_1)
        ratio     = cd_encode / (cd_decode + 1e-8)

        results.append(dict(start=start, cd_encode=cd_encode, cd_decode=cd_decode, ratio=ratio))
        print(f"  {start:6d}  {cd_encode:12.4f}  {cd_decode:12.4f}  {ratio:8.2f}x")

    print()
    mean_enc = np.mean([r["cd_encode"] for r in results])
    mean_dec = np.mean([r["cd_decode"] for r in results])
    print(f"  {'mean':>6}  {mean_enc:12.4f}  {mean_dec:12.4f}  {mean_enc/(mean_dec+1e-8):8.2f}x")
    print()
    if mean_enc > 2 * mean_dec:
        print("  → encode variance dominates: z_star distribution is meaningfully multi-modal.")
        print("    Flow matching would model this distribution rather than averaging over it.")
    elif mean_enc > mean_dec:
        print("  → encode variance is larger than decode noise, but not dramatically so.")
        print("    Flow matching may help at the margin.")
    else:
        print("  → encode variance ≈ decode noise floor.")
        print("    z_star is effectively deterministic; flow matching adds little over regression.")


if __name__ == "__main__":
    main()
