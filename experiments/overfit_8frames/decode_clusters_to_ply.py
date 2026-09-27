#!/usr/bin/env python3
"""
Decode NOVA3R room0 cluster centres to a world-frame point cloud.

Loads the 768 K-means cluster centres from the room0 z_star clustering, then
decodes them N times — once per cached training window — using each window's
own pts_norm as the flow-matching conditioning input.  Every decode is
un-normalised and transformed to world frame; the results are merged into a
single PLY covering the full room.

Usage (run on the cluster with CUDA):
  ! python experiments/overfit_8frames/decode_clusters_to_ply.py
  ! python experiments/overfit_8frames/decode_clusters_to_ply.py --num-queries 4096 --max-samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from omegaconf import OmegaConf

REPO_ROOT   = Path(__file__).resolve().parents[2]
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.inference import normalize_input, amp_dtype_mapping  # noqa: E402
from multi_scene_train import (                                   # noqa: E402
    crop_frustum_world_points,
    sample_points,
    world_to_first_camera,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=Path,
                   default=Path(__file__).parent / "config.yaml")
    p.add_argument("--clusters",    type=Path,
                   default=REPO_ROOT / "outputs" / "latent_clusters"
                           / "room0_N1845_zstar_k768" / "centers.npy")
    p.add_argument("--cache-dir",   type=Path,
                   default=Path(__file__).parent / "data" / "multi_N50_s20")
    p.add_argument("--out-dir",     type=Path,
                   default=REPO_ROOT / "outputs" / "cluster_decode")
    p.add_argument("--num-queries", type=int, default=8192)
    p.add_argument("--max-samples", type=int, default=50,
                   help="Maximum number of cached windows to decode (default: all)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--replica-root", type=Path, default=None,
                   help="Override replica_root from config (useful for local runs)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def height_cmap(pts: np.ndarray, cmap: str = "plasma") -> np.ndarray:
    y = pts[:, 1]
    norm = Normalize(np.percentile(y, 2), np.percentile(y, 98))
    return plt.get_cmap(cmap)(norm(y))[:, :3].astype(np.float32)


def save_ply(path: Path, pts: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor,    # (1, 768, 128)
                  pts_norm: torch.Tensor,  # (1, 8192, 3)
                  device: torch.device,
                  num_queries: int, seed: int) -> np.ndarray:
    """Decode tokens → point cloud in normalised space (median ≈ 3).
    Returns (num_queries, 3) float32."""
    torch.manual_seed(seed)
    B = 1
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(B, num_queries, 3, device=device) * 2 - 1
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
    pts3d = (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()
    return pts3d  # (num_queries, 3)


def pts_norm_and_factor(pts_cam: np.ndarray,
                        device: torch.device) -> tuple[torch.Tensor, float]:
    """Normalise pts_cam (N, 3) → pts_norm (1, N, 3) with median_3 mode.
    Returns (pts_norm, norm_factor)."""
    pts_t  = torch.from_numpy(pts_cam).unsqueeze(0).float().to(device)  # (1, N, 3)
    valid  = torch.ones(1, pts_cam.shape[0], dtype=torch.bool, device=device)
    pts_n, _ = normalize_input(pts_t, valid, pts_t, valid, mode="median_3")
    norm_factor = float(torch.from_numpy(pts_cam).norm(dim=-1).median().clamp(0.01, 100.0))
    return pts_n.cpu(), norm_factor


def main() -> None:
    args = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)

    # ── cluster centres ────────────────────────────────────────────────────────
    centers_np = np.load(args.clusters).astype(np.float32)  # (768, 128)
    tokens = torch.from_numpy(centers_np).unsqueeze(0)      # (1, 768, 128)
    print(f"[decode] cluster centres: {centers_np.shape}  from {args.clusters}")

    # ── NOVA3R AE ──────────────────────────────────────────────────────────────
    # Remap Linux path to local path if needed
    nova3r_ckpt_raw = str(cfg.nova3r_ckpt)
    nova3r_ckpt_local = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova3r_ckpt = nova3r_ckpt_local if not Path(nova3r_ckpt_raw).exists() else nova3r_ckpt_raw
    print(f"[decode] loading NOVA3R from {nova3r_ckpt} ...")
    nova_model, nova_cfg = load_nova3r_model(nova3r_ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── mesh for computing pts_cam (conditioning input) ────────────────────────
    replica_root = args.replica_root if args.replica_root is not None else Path(str(cfg.replica_root))
    room         = str(cfg.room)
    mesh_ply     = replica_root / f"{room}_mesh.ply"
    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = torch.tensor([[cam_p["fx"], 0., cam_p["cx"]],
                          [0., cam_p["fy"], cam_p["cy"]],
                          [0., 0., 1.]], dtype=torch.float32)

    print(f"[decode] loading mesh from {mesh_ply} ...")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))
    print(f"[decode] mesh: {len(mesh_pts):,} sampled points")

    # ── cached windows ─────────────────────────────────────────────────────────
    sample_dirs = sorted(args.cache_dir.glob("sample_*"))
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_* dirs found in {args.cache_dir}")
    sample_dirs = sample_dirs[: args.max_samples]
    print(f"[decode] found {len(sample_dirs)} cached windows (using {len(sample_dirs)})")

    # ── decode loop ────────────────────────────────────────────────────────────
    all_pts_world: list[np.ndarray] = []
    n_frames = int(cfg.num_frames)

    for si, sample_dir in enumerate(sample_dirs):
        meta = torch.load(sample_dir / "meta.pt", map_location="cpu", weights_only=False)
        if meta["room"] != room:
            continue

        poses_c2w = meta["poses_c2w"].float()  # (8, 4, 4)
        first_c2w = poses_c2w[0]               # (4, 4) world frame, metres

        # Build pts_cam: mesh points from all 8 frusta in first-camera frame
        pts_cam_list = []
        for i in range(n_frames):
            frust = crop_frustum_world_points(
                mesh_pts, poses_c2w[i], K_nat, (H_nat, W_nat)
            )
            pts_cam_list.append(world_to_first_camera(frust, first_c2w))
        all_frust = torch.cat(pts_cam_list, dim=0)
        pts_cam   = sample_points(all_frust, int(cfg.mesh_sample_points),
                                  seed=meta["start_idx"]).numpy()  # (8192, 3)

        pts_norm_t, norm_factor = pts_norm_and_factor(pts_cam, device)  # (1,8192,3), float

        pts3d_norm = decode_tokens(
            nova_model, nova_cfg,
            tokens, pts_norm_t,
            device, args.num_queries,
            seed=args.seed + si,
        )  # (num_queries, 3) normalised

        # Un-normalise: pts_cam = pts3d_norm / 3 * norm_factor
        pts_cam_dec = pts3d_norm / 3.0 * norm_factor  # (N, 3) camera frame, metres

        # Camera frame → world frame
        ones      = np.ones((len(pts_cam_dec), 1), dtype=np.float32)
        c2w_np    = first_c2w.numpy()
        pts_world = (c2w_np @ np.hstack([pts_cam_dec, ones]).T).T[:, :3]
        all_pts_world.append(pts_world.astype(np.float32))

        print(f"  sample {si:03d} | start={meta['start_idx']:4d} "
              f"| norm_factor={norm_factor:.3f}m "
              f"| decoded {len(pts_world):,} pts")

    # ── merge and save ─────────────────────────────────────────────────────────
    all_pts = np.concatenate(all_pts_world, axis=0)  # (N_total, 3)
    colors  = height_cmap(all_pts, cmap="plasma")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = args.out_dir / f"room0_cluster_decode_N{len(sample_dirs)}_q{args.num_queries}.ply"
    save_ply(out_ply, all_pts, colors)
    print(f"\n[decode] {len(all_pts):,} total points saved → {out_ply}")

    # Also save a thumbnail scatter
    out_png = out_ply.with_suffix(".png")
    step = max(1, len(all_pts) // 40_000)
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(all_pts[::step, 0], all_pts[::step, 1], all_pts[::step, 2],
               c=colors[::step], s=1.5, linewidths=0, alpha=0.7, depthshade=False)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Room0 cluster-centre decode  ({len(all_pts):,} pts)", fontsize=10)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[decode] thumbnail → {out_png}")


if __name__ == "__main__":
    main()
