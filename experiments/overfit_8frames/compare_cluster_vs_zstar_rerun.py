#!/usr/bin/env python3
"""
Compare cluster-centre decode vs. real z_star decode in Rerun.

For each cached window, decodes twice using the same pts_norm conditioning
and the same norm_factor / c2w transform:
  - orange  : 768 K-means cluster centres (aggregate room0 representation)
  - blue    : real z_star tokens for that window (control)

Both point clouds are in world frame, covering the full room trajectory.

Usage:
  python experiments/overfit_8frames/compare_cluster_vs_zstar_rerun.py
  python experiments/overfit_8frames/compare_cluster_vs_zstar_rerun.py --max-samples 10
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
    p.add_argument("--config",       type=Path,
                   default=Path(__file__).parent / "config.yaml")
    p.add_argument("--clusters",     type=Path,
                   default=REPO_ROOT / "outputs" / "latent_clusters"
                           / "room0_N1845_zstar_k768" / "centers.npy")
    p.add_argument("--cache-dir",    type=Path,
                   default=Path(__file__).parent / "data" / "multi_N50_s20")
    p.add_argument("--replica-root", type=Path, default=None)
    p.add_argument("--num-queries",  type=int, default=8192)
    p.add_argument("--max-samples",  type=int, default=50)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--rrd-out",      type=Path,
                   default=REPO_ROOT / "outputs" / "cluster_decode"
                           / "cluster_vs_zstar.rrd",
                   help="Save .rrd file here instead of spawning live viewer")
    p.add_argument("--spawn",        action="store_true",
                   help="Spawn the Rerun viewer immediately (default: save .rrd)")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor,    # (1, 768, 128)
                  pts_norm: torch.Tensor,  # (1, 8192, 3)
                  device: torch.device,
                  num_queries: int, seed: int) -> np.ndarray:
    """Returns decoded pts in normalised space, shape (num_queries, 3)."""
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


def norm_to_world(pts3d_norm: np.ndarray, norm_factor: float,
                  c2w: np.ndarray) -> np.ndarray:
    """Un-normalise (median_3) and transform to world frame."""
    pts_cam = pts3d_norm / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


def main() -> None:
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)

    replica_root = (args.replica_root
                    if args.replica_root is not None
                    else Path(str(cfg.replica_root)))
    room = str(cfg.room)

    # ── cluster centres ────────────────────────────────────────────────────────
    centers_np = np.load(args.clusters).astype(np.float32)  # (768, 128)
    cluster_tokens = torch.from_numpy(centers_np).unsqueeze(0)  # (1, 768, 128)
    print(f"[compare] cluster centres: {centers_np.shape}")

    # ── NOVA3R ─────────────────────────────────────────────────────────────────
    ckpt_raw   = str(cfg.nova3r_ckpt)
    ckpt_local = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    ckpt_path  = ckpt_local if not Path(ckpt_raw).exists() else ckpt_raw
    print(f"[compare] loading NOVA3R from {ckpt_path} ...")
    nova_model, nova_cfg = load_nova3r_model(ckpt_path, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── mesh ───────────────────────────────────────────────────────────────────
    mesh_ply = replica_root / f"{room}_mesh.ply"
    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = torch.tensor([[cam_p["fx"], 0., cam_p["cx"]],
                          [0., cam_p["fy"], cam_p["cy"]],
                          [0., 0., 1.]], dtype=torch.float32)
    print(f"[compare] loading mesh ...")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    # ── cached windows ─────────────────────────────────────────────────────────
    sample_dirs = sorted(args.cache_dir.glob("sample_*"))[: args.max_samples]
    n_frames    = int(cfg.num_frames)
    print(f"[compare] decoding {len(sample_dirs)} windows ...")

    cluster_world_pts: list[np.ndarray] = []
    zstar_world_pts:   list[np.ndarray] = []

    for si, sample_dir in enumerate(sample_dirs):
        meta = torch.load(sample_dir / "meta.pt", map_location="cpu", weights_only=False)
        if meta["room"] != room:
            continue

        z_star = torch.load(sample_dir / "z_star.pt",
                            map_location="cpu", weights_only=True)  # (1, 768, 128)

        poses_c2w = meta["poses_c2w"].float()
        first_c2w = poses_c2w[0]

        # Recompute pts_cam from mesh to get correct norm_factor
        pts_cam_list = []
        for i in range(n_frames):
            frust = crop_frustum_world_points(
                mesh_pts, poses_c2w[i], K_nat, (H_nat, W_nat))
            pts_cam_list.append(world_to_first_camera(frust, first_c2w))
        pts_cam = sample_points(
            torch.cat(pts_cam_list, dim=0),
            int(cfg.mesh_sample_points),
            seed=meta["start_idx"],
        ).numpy()  # (8192, 3)

        norm_factor = float(
            torch.from_numpy(pts_cam).norm(dim=-1).median().clamp(0.01, 100.0))
        pts_t   = torch.from_numpy(pts_cam).unsqueeze(0).float().to(device)
        valid_t = torch.ones(1, pts_cam.shape[0], dtype=torch.bool, device=device)
        pts_norm, _ = normalize_input(pts_t, valid_t, pts_t, valid_t, mode="median_3")
        pts_norm_cpu = pts_norm.cpu()

        c2w_np = first_c2w.numpy()

        # Cluster-centre decode
        pts3d_cluster = decode_tokens(
            nova_model, nova_cfg, cluster_tokens, pts_norm_cpu,
            device, args.num_queries, seed=args.seed + si,
        )
        cluster_world_pts.append(norm_to_world(pts3d_cluster, norm_factor, c2w_np))

        # Real z_star decode (control)
        pts3d_zstar = decode_tokens(
            nova_model, nova_cfg, z_star, pts_norm_cpu,
            device, args.num_queries, seed=args.seed + si,
        )
        zstar_world_pts.append(norm_to_world(pts3d_zstar, norm_factor, c2w_np))

        print(f"  sample {si:03d} | start={meta['start_idx']:4d} "
              f"| norm_factor={norm_factor:.3f}m", flush=True)

    cluster_all = np.concatenate(cluster_world_pts, axis=0)
    zstar_all   = np.concatenate(zstar_world_pts,   axis=0)
    print(f"\n[compare] cluster decode: {len(cluster_all):,} pts")
    print(f"[compare] z_star control: {len(zstar_all):,} pts")

    # ── Rerun ──────────────────────────────────────────────────────────────────
    import rerun as rr

    rr.init("cluster_vs_zstar", spawn=args.spawn)
    if not args.spawn:
        args.rrd_out.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(args.rrd_out))

    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Subsample for viewer performance
    def subsample(pts: np.ndarray, n: int = 200_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.choice(len(pts), n, replace=False)]

    ORANGE = np.array([255, 140,  30], dtype=np.uint8)
    BLUE   = np.array([ 50, 150, 255], dtype=np.uint8)

    rr.log(
        "world/cluster_decode",
        rr.Points3D(
            subsample(cluster_all),
            colors=np.tile(ORANGE, (min(len(cluster_all), 200_000), 1)),
            radii=0.012,
        ),
    )
    rr.log(
        "world/zstar_control",
        rr.Points3D(
            subsample(zstar_all),
            colors=np.tile(BLUE, (min(len(zstar_all), 200_000), 1)),
            radii=0.012,
        ),
    )
    rr.log(
        "legend",
        rr.TextDocument(
            "# Cluster vs z_star decode\n\n"
            "- **Orange** `world/cluster_decode`: 768 K-means cluster centres\n"
            "- **Blue** `world/zstar_control`: real z_star tokens (per-window GT)\n\n"
            f"Each layer: {len(sample_dirs)} windows × {args.num_queries:,} pts = "
            f"{len(sample_dirs)*args.num_queries:,} total",
            media_type=rr.MediaType.MARKDOWN,
        ),
    )

    if args.spawn:
        print("[compare] Rerun viewer launched.")
    else:
        print(f"[compare] saved → {args.rrd_out}")
        print(f"  open with:  rerun {args.rrd_out}")


if __name__ == "__main__":
    main()
