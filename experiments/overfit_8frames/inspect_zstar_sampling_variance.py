#!/usr/bin/env python3
"""
Measure z* sampling variance for a fixed 8-frame window.

Encodes the same concatenated visible point cloud N times with different
random subsamples (different seeds) and reports per-slot and pairwise
statistics. This separates intrinsic encoder noise from inter-window
variance.

Usage:
    python inspect_zstar_sampling_variance.py [--config config.yaml] [--num-seeds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P = NOVA3R_ROOT / "third_party"
DA3_SRC = REPO_ROOT / "da3" / "src"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model  # noqa: E402
from nova3r.inference import normalize_input  # noqa: E402
from vc3r.replica import crop_visible_world_points  # noqa: E402


def world_to_first_camera(pts_world: torch.Tensor, first_pose_c2w: torch.Tensor) -> torch.Tensor:
    w2c = torch.linalg.inv(first_pose_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


def sample_points(pts: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if pts.shape[0] >= count:
        idx = torch.randperm(pts.shape[0], generator=gen)[:count]
    else:
        pad = torch.randint(pts.shape[0], (count - pts.shape[0],), generator=gen)
        idx = torch.cat([torch.arange(pts.shape[0]), pad])
    return pts[idx]


@torch.no_grad()
def encode_nova3r_ae(
    model,
    cfg,
    pts_first_cam: torch.Tensor,  # (1, N, 3)
    device: torch.device,
) -> torch.Tensor:
    """Returns z_star (1, K, D)."""
    pts = pts_first_cam.to(device).float()
    valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)
    norm_mode = cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
    encoder_data = model._encode(pointmaps=pts_norm, test=True)
    return encoder_data["tokens"].float().cpu()


def summarize(name: str, values: torch.Tensor) -> None:
    v = values.float().flatten()
    print(
        f"  {name}: "
        f"min={v.min().item():.6f}  max={v.max().item():.6f}  "
        f"mean={v.mean().item():.6f}  std={v.std(unbiased=False).item():.6f}  "
        f"median={v.median().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--num-seeds", type=int, default=10,
                        help="Number of disjoint training slices; validation uses slot num_seeds")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--replica-root", type=Path, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.stride is not None:
        cfg.frame_stride = args.stride
    device = torch.device(args.device)

    stride = int(cfg.frame_stride)
    n_frames = int(cfg.num_frames)
    replica_root = (args.replica_root if args.replica_root is not None
                    else Path(str(cfg.replica_root)))
    room_dir = replica_root / cfg.room
    results_dir = room_dir / "results"
    mesh_ply = replica_root / f"{cfg.room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]

    depth_scale = float(cam["scale"])
    H_native = int(cam["h"])
    W_native = int(cam["w"])
    K_native = torch.tensor([
        [cam["fx"], 0.0, cam["cx"]],
        [0.0, cam["fy"], cam["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]

    sampled_ids = [frame_ids[i * stride] for i in range(n_frames)]
    sampled_poses = torch.from_numpy(
        np.stack([poses_all[fid] for fid in sampled_ids])
    ).float()

    print(f"room:    {cfg.room}")
    print(f"frames:  {sampled_ids}  (stride={stride})")
    print(f"device:  {device}")
    print(f"seeds:   0 .. {args.num_seeds - 1}")
    print(f"sample_n: {cfg.mesh_sample_points}")

    def load_depth(fid: int) -> torch.Tensor:
        depth_img = Image.open(results_dir / f"depth{fid:06d}.png")
        return torch.from_numpy(np.asarray(depth_img, dtype=np.float32) / depth_scale)

    print(f"\nLoading mesh: {mesh_ply}")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))
    print(f"Mesh surface points: {mesh_pts.shape[0]:,}")

    first_pose_c2w = sampled_poses[0]
    all_visible_list: list[torch.Tensor] = []
    for i, fid in enumerate(sampled_ids):
        depth = load_depth(fid)
        c2w = sampled_poses[i]
        visible = crop_visible_world_points(
            points_world=mesh_pts,
            camera_to_world=c2w,
            intrinsics=K_native,
            depth=depth,
            depth_tolerance=args.depth_tolerance,
        )
        pts_cam = world_to_first_camera(visible["points_world"], first_pose_c2w)
        all_visible_list.append(pts_cam)
        print(f"  frame {fid:06d}: {pts_cam.shape[0]:>6} pts in first-cam frame")

    all_visible = torch.cat(all_visible_list, dim=0)
    n_total = all_visible.shape[0]
    k = int(cfg.mesh_sample_points)
    if n_total < 2 * k:
        raise RuntimeError(
            f"Not enough visible points ({n_total:,}) to reserve a disjoint validation slice of {k} pts. "
            f"Need at least {2 * k:,}."
        )

    # Shuffle once with a fixed base seed.
    # The last k points are reserved as the validation set — no training sample can touch them.
    # Training samples draw independently (with possible overlap) from the first N-k points.
    base_gen = torch.Generator().manual_seed(0xDEADBEEF)
    shuffle_idx = torch.randperm(n_total, generator=base_gen)
    all_visible_shuffled = all_visible[shuffle_idx]

    train_pool = all_visible_shuffled[:-k]   # (N-k, 3)
    val_pts    = all_visible_shuffled[-k:]   # (k, 3)  — strictly disjoint
    print(f"Total visible pts: {n_total:,}  sample_n={k}  train_pool={len(train_pool):,}  val_pts={len(val_pts):,}")

    print("Loading NOVA3R AE ...")
    ckpt_local = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova3r_model, nova3r_cfg = load_nova3r_model(ckpt_local, str(device))
    nova3r_model.eval()
    for p in nova3r_model.parameters():
        p.requires_grad_(False)

    zstars: list[torch.Tensor] = []
    for seed in range(args.num_seeds):
        pts = sample_points(train_pool, k, seed=seed)
        z = encode_nova3r_ae(nova3r_model, nova3r_cfg, pts.unsqueeze(0), device)
        z = z[0]  # (768, 128)
        zstars.append(z)
        print(
            f"  seed={seed:02d}  shape={tuple(z.shape)}  "
            f"mean={z.mean().item():.6f}  std={z.std(unbiased=False).item():.6f}  "
            f"l2={z.norm().item():.4f}"
        )

    stacked = torch.stack(zstars)  # (S, 768, 128)
    mean_zstar = stacked.mean(dim=0)  # (768, 128)

    # ── Test 1: z* variance within the training pool ──────────────────────────
    print("\n--- test 1: intra-pool variance (training seeds) ---")
    per_slot_std = stacked.std(dim=0, unbiased=False)        # (768, 128)
    per_slot_mean_std = per_slot_std.mean(dim=-1)            # (768,)
    summarize("slot_std (mean over feat_dim)", per_slot_mean_std)
    summarize("slot_std (element-wise)",       per_slot_std)

    per_slot_mse_to_mean = F.mse_loss(
        stacked, mean_zstar.unsqueeze(0).expand_as(stacked), reduction="none"
    ).mean(dim=-1)                                           # (S, 768)
    summarize("per_slot_mse_to_mean", per_slot_mse_to_mean)

    mse_values: list[float] = []
    cosine_values: list[float] = []
    for i in range(args.num_seeds):
        for j in range(i + 1, args.num_seeds):
            mse_values.append(F.mse_loss(zstars[i], zstars[j]).item())
            cosine_values.append(
                F.cosine_similarity(zstars[i].flatten(), zstars[j].flatten(), dim=0).item()
            )
    summarize("pairwise_mse",    torch.tensor(mse_values))
    summarize("pairwise_cosine", torch.tensor(cosine_values))
    intra_mse = torch.tensor(mse_values).mean().item()

    # ── Test 2: validation set vs training distribution ───────────────────────
    print(f"\n--- test 2: validation set vs training distribution ---")
    val_z = encode_nova3r_ae(nova3r_model, nova3r_cfg, val_pts.unsqueeze(0), device)[0]
    val_mse_vs_mean = F.mse_loss(val_z, mean_zstar).item()
    val_cos_vs_mean = F.cosine_similarity(val_z.flatten(), mean_zstar.flatten(), dim=0).item()
    val_mse_per_seed = [F.mse_loss(val_z, z).item() for z in zstars]
    print(f"  mse vs training mean:    {val_mse_vs_mean:.6f}")
    print(f"  cosine vs training mean: {val_cos_vs_mean:.6f}")
    summarize("mse vs each training seed", torch.tensor(val_mse_per_seed))

    print("\n--- summary ---")
    print(f"  intra-pool mean pairwise MSE (test 1): {intra_mse:.6f}")
    print(f"  val vs training mean MSE   (test 2): {val_mse_vs_mean:.6f}")


if __name__ == "__main__":
    main()