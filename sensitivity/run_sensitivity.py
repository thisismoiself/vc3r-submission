#!/usr/bin/env python3
"""
Sensitivity analysis for z_star dimensions.

For a single cached window:
  1. Measures per-dimension decoder sensitivity: perturb dimension d across all
     768 tokens by one global std, decode, measure Chamfer distance to baseline.
  2. Measures per-dimension seed noise: re-encode geometry N_SEEDS times,
     align each seed's tokens to the consensus centroids via Hungarian matching,
     compute per-dimension variance of the residuals.
  3. Computes SNR = sensitivity / noise_var and derives loss weights.

Outputs (saved to sensitivity/val<N>/):
  sensitivity.pt   (128,)  Chamfer distance per dim after ε-perturbation
  noise_var.pt     (128,)  Mean per-dim variance across aligned seeds
  snr.pt           (128,)  sensitivity / (noise_var + eps)
  weights.pt       (128,)  SNR normalised so mean = 1
  scatter.png              sensitivity vs noise_var scatter (quadrant view)
  snr_bar.png              SNR sorted descending (elbow view)

Usage:
  ! python sensitivity/run_sensitivity.py
  ! python sensitivity/run_sensitivity.py --val-start 56 --replica-root datasets/replica
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import linear_sum_assignment

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model              # noqa: E402
from nova3r.inference import normalize_input, amp_dtype_mapping      # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper            # noqa: E402
from nova3r.flow_matching.solver import ODESolver                    # noqa: E402
from vc3r.replica import crop_visible_world_points  # noqa: E402

DATA_ROOT  = REPO_ROOT / "scripts" / "data" / "windows"
OUT_ROOT   = REPO_ROOT / "sensitivity"

N_FRAMES   = 8
K          = 8192
N_CLUSTERS = 768


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-start",    type=int,  default=56)
    p.add_argument("--replica-root", type=Path, default=None)
    p.add_argument("--room",         default="room0")
    p.add_argument("--n-seeds",      type=int,  default=20)
    p.add_argument("--num-queries",  type=int,  default=8192)
    p.add_argument("--decode-seed",  type=int,  default=42)
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_window(start: int):
    win_dir   = DATA_ROOT / f"start_{start:04d}"
    meta      = torch.load(win_dir / "meta.pt",             map_location="cpu", weights_only=False)
    consensus = torch.load(win_dir / "z_star_consensus.pt", map_location="cpu", weights_only=True)
    pts_norm  = torch.load(win_dir / "pts_norm.pt",         map_location="cpu", weights_only=True)
    return meta, consensus, pts_norm


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor, pts_norm: torch.Tensor,
                  device: torch.device, num_queries: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, num_queries, 3, device=device) * 2 - 1
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


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    dists  = torch.cdist(a_t.unsqueeze(0), b_t.unsqueeze(0))[0]  # (N, M) L2
    a_to_b = dists.min(dim=1).values.mean()
    b_to_a = dists.min(dim=0).values.mean()
    return float((a_to_b + b_to_a) / 2)


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
def encode_once(model, norm_mode: str, pts: torch.Tensor, device: torch.device) -> torch.Tensor:
    pts_dev = pts.unsqueeze(0).to(device).float()
    valid   = torch.ones(pts_dev.shape[:2], dtype=torch.bool, device=device)
    pts_norm, _ = normalize_input(pts_dev, valid, pts_dev, valid, mode=norm_mode)
    tokens  = model._encode(pointmaps=pts_norm, test=True)["tokens"].float().cpu()
    return tokens[0]  # (768, 128)


def hungarian_align(z_seed: torch.Tensor, z_consensus: torch.Tensor) -> torch.Tensor:
    """Reorder z_seed rows to minimise total L2 distance to z_consensus rows."""
    cost = torch.cdist(z_consensus, z_seed).numpy()  # (768, 768)
    _, col_ind = linear_sum_assignment(cost)
    return z_seed[col_ind]


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    replica_root = (args.replica_root if args.replica_root is not None
                    else REPO_ROOT / "datasets" / "replica")
    room_dir    = replica_root / args.room
    results_dir = room_dir / "results"
    mesh_ply    = replica_root / f"{args.room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    depth_scale = float(cam["scale"])
    K_native    = torch.tensor([
        [cam["fx"], 0., cam["cx"]],
        [0., cam["fy"], cam["cy"]],
        [0., 0., 1.]], dtype=torch.float32)

    poses_all     = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files   = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]

    # ── load cached val window ────────────────────────────────────────────────
    print(f"[load] val window start={args.val_start}")
    meta, z_consensus, pts_norm = load_window(args.val_start)
    z_ref            = z_consensus          # (1, 768, 128)
    z_consensus_toks = z_consensus[0]       # (768, 128)

    # ── load NOVA3R ───────────────────────────────────────────────────────────
    print("[model] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    # ── Step 1: baseline decode ───────────────────────────────────────────────
    print("[step1] Baseline decode …")
    pts_ref = decode_tokens(nova_model, nova_cfg, z_ref, pts_norm,
                            device, args.num_queries, seed=args.decode_seed)
    eps = float(z_ref.std())
    print(f"  ε = {eps:.5f}  (global std of token embeddings)")

    # ── Step 2: per-dimension sensitivity ────────────────────────────────────
    print("[step2] Sensitivity — 128 decode passes …")
    sensitivity = np.zeros(128, dtype=np.float32)
    for d in range(128):
        z_pert = z_ref.clone()
        z_pert[:, :, d] += eps
        pts_pert      = decode_tokens(nova_model, nova_cfg, z_pert, pts_norm,
                                      device, args.num_queries, seed=args.decode_seed)
        sensitivity[d] = chamfer_distance(pts_pert, pts_ref)
        if (d + 1) % 16 == 0:
            print(f"  dim {d+1:3d}/128  last sensitivity={sensitivity[d]:.5f}", flush=True)

    # ── Step 3: per-dimension seed variance ───────────────────────────────────
    print("[step3] Building visible point pool …")

    def load_depth(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"depth{fid:06d}.png")
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / depth_scale)

    frame_ids = meta["frame_ids"]
    poses_c2w = meta["poses_c2w"]
    first_c2w = meta["first_c2w"]

    print("  Loading mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    pool_list: list[torch.Tensor] = []
    for i, fid in enumerate(frame_ids):
        depth = load_depth(fid)
        vis   = crop_visible_world_points(
            points_world=mesh_pts,
            camera_to_world=poses_c2w[i],
            intrinsics=K_native,
            depth=depth,
            depth_tolerance=args.depth_tolerance,
        )
        pool_list.append(world_to_first_camera(vis["points_world"], first_c2w))
    pool = torch.cat(pool_list, dim=0)
    print(f"  pool: {len(pool):,} pts")

    print(f"[step3] Encoding {args.n_seeds} seeds + Hungarian alignment …")
    residuals: list[torch.Tensor] = []
    for seed in range(args.n_seeds):
        pts      = sample_pts(pool, K, seed=seed)
        z_seed   = encode_once(nova_model, norm_mode, pts, device)      # (768, 128)
        z_aligned = hungarian_align(z_seed, z_consensus_toks)           # (768, 128)
        residuals.append(z_aligned - z_consensus_toks)
        print(f"  seed {seed+1:2d}/{args.n_seeds}", flush=True)

    R         = torch.stack(residuals)           # (N_SEEDS, 768, 128)
    noise_var = R.var(dim=0).mean(dim=0).numpy() # (128,) — mean token variance per dim

    # ── Step 4: SNR and weights ───────────────────────────────────────────────
    _eps    = 1e-6
    snr     = sensitivity / (noise_var + _eps)
    weights = snr / (snr.sum() + _eps) * 128    # normalised so mean = 1

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = OUT_ROOT / f"val{args.val_start}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(torch.from_numpy(sensitivity), out_dir / "sensitivity.pt")
    torch.save(torch.from_numpy(noise_var),   out_dir / "noise_var.pt")
    torch.save(torch.from_numpy(snr),         out_dir / "snr.pt")
    torch.save(torch.from_numpy(weights),     out_dir / "weights.pt")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[results]")
    print(f"  sensitivity  min={sensitivity.min():.5f}  max={sensitivity.max():.5f}  "
          f"mean={sensitivity.mean():.5f}")
    print(f"  noise_var    min={noise_var.min():.5f}  max={noise_var.max():.5f}  "
          f"mean={noise_var.mean():.5f}")
    print(f"  snr          min={snr.min():.5f}  max={snr.max():.5f}  "
          f"mean={snr.mean():.5f}")

    top10 = np.argsort(snr)[::-1][:10]
    bot10 = np.argsort(snr)[:10]
    print(f"  top-10 dims by SNR: {top10.tolist()}")
    print(f"  bot-10 dims by SNR: {bot10.tolist()}")

    n_useful = int((snr > snr.mean()).sum())
    print(f"  dims above mean SNR: {n_useful}/128")

    # ── Plots ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    med_noise = float(np.median(noise_var))
    med_sens  = float(np.median(sensitivity))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(noise_var, sensitivity, s=18, alpha=0.75, color="steelblue", zorder=3)
    ax.axvline(med_noise, color="grey", linestyle="--", linewidth=0.8, label="median noise_var")
    ax.axhline(med_sens,  color="grey", linestyle=":",  linewidth=0.8, label="median sensitivity")
    for i in [top10[0], top10[1], top10[2]]:
        ax.annotate(str(i), (noise_var[i], sensitivity[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="navy")
    ax.set_xlabel("Noise variance (mean across 768 tokens)")
    ax.set_ylabel("Sensitivity (Chamfer distance after ε-perturbation)")
    ax.set_title(f"z_star dimension sensitivity vs noise — val start={args.val_start}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter.png", dpi=150)
    plt.close(fig)

    snr_sorted = np.sort(snr)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(128), snr_sorted, color="steelblue", width=1.0)
    ax.set_xlabel("Dimension rank (by SNR, descending)")
    ax.set_ylabel("SNR (sensitivity / noise_var)")
    ax.set_title(f"z_star SNR profile — val start={args.val_start}")
    fig.tight_layout()
    fig.savefig(out_dir / "snr_bar.png", dpi=150)
    plt.close(fig)

    print(f"\n[done] → {out_dir}/")
    print(f"  sensitivity.pt  noise_var.pt  snr.pt  weights.pt")
    print(f"  scatter.png  snr_bar.png")


if __name__ == "__main__":
    main()
