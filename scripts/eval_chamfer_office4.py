#!/usr/bin/env python3
"""Evaluate adapter vs. a fresh single-seed NOVA3R encode using Chamfer distance.

For each validation window the script:
  1. Builds the visible-mesh point pool from the Replica scene (identical to caching).
  2. Samples K=8192 points with seed=0 and runs NOVA3R encode_once → ground-truth z_star.
  3. Extracts DA3 tokens from the scene RGB images.
  4. Runs the adapter on those DA3 tokens → predicted z_star.
  5. Decodes both z_stars into point clouds via NOVA3R ODE.
  6. Converts both clouds to world frame and computes symmetric Chamfer distance.

Fixed evaluation protocol: 10 windows, 8 frames each at stride 20, starting at
0, 200, 400, …, 1800. Override with --starts / --frame-stride.

Usage:
    python eval_chamfer_office4.py \\
        --checkpoint path/to/best.pt \\
        --replica-root path/to/replica \\
        --room office4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P = NOVA3R_ROOT / "third_party"
DA3_SRC = REPO_ROOT / "da3" / "src"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from vc3r.replica import crop_visible_world_points  # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper  # noqa: E402
from nova3r.flow_matching.solver import ODESolver  # noqa: E402
from nova3r.inference import normalize_input  # noqa: E402
from multi_scene_train import load_da3_model, extract_da3_tokens  # noqa: E402

NOVA3R_CKPT = REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth"
DEFAULT_REPLICA = REPO_ROOT / "datasets" / "replica"
K = 8192
N_FRAMES = 8
GT_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Adapter checkpoint (.pt) to evaluate.")
    p.add_argument("--replica-root", type=Path, default=DEFAULT_REPLICA)
    p.add_argument("--room", default="office4",
                   help="Replica scene to evaluate on.")
    p.add_argument("--starts", type=int, nargs="+", default=list(range(0, 2000, 200)),
                   help="Window start frame indices (default: 0, 200, …, 1800).")
    p.add_argument("--frame-stride", type=int, default=20,
                   help="Frame stride within each window (default: 20).")
    p.add_argument("--da3-model", default="depth-anything/DA3-LARGE-1.1")
    p.add_argument("--da3-layer", type=int, nargs="+", default=[3],
                   help="DA3 backbone layer(s) to extract. Multiple layers are concatenated.")
    p.add_argument("--da3-max-tokens", type=int, default=2048,
                   help="Max tokens to keep per DA3 layer.")
    p.add_argument("--image-height", type=int, default=392)
    p.add_argument("--image-width", type=int, default=518)
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--num-queries", type=int, default=8192,
                   help="Number of points to sample when decoding z_star.")
    p.add_argument("--subsample", type=int, default=10_000,
                   help="Max points per cloud when computing Chamfer distance.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for NOVA3R ODE sampling.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def adapter_config_from_state_dict(sd: dict) -> dict:
    target_tokens, hidden_dim = sd["target_queries"].shape
    source_dim = sd["source_proj.weight"].shape[1]
    target_dim = sd["out_proj.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".query_norm.weight"))
    num_heads = hidden_dim // 64
    return dict(source_dim=source_dim, hidden_dim=hidden_dim,
                target_tokens=target_tokens, target_dim=target_dim,
                depth=depth, num_heads=num_heads)


def sample_pts(pool: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if pool.shape[0] >= k:
        idx = torch.randperm(pool.shape[0], generator=gen)[:k]
    else:
        pad = torch.randint(pool.shape[0], (k - pool.shape[0],), generator=gen)
        idx = torch.cat([torch.arange(pool.shape[0]), pad])
    return pool[idx]


def world_to_first_camera(pts_world: torch.Tensor, first_c2w: torch.Tensor) -> torch.Tensor:
    w2c = torch.linalg.inv(first_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


@torch.no_grad()
def encode_once(model, norm_mode: str, pts: torch.Tensor, device: torch.device):
    pts_dev = pts.unsqueeze(0).to(device).float()
    valid = torch.ones(pts_dev.shape[:2], dtype=torch.bool, device=device)
    pts_norm, _ = normalize_input(pts_dev, valid, pts_dev, valid, mode=norm_mode)
    tokens = model._encode(pointmaps=pts_norm, test=True)["tokens"].float().cpu()
    nf = float(pts_dev.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))
    return tokens[0], pts_norm.cpu(), nf


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens: torch.Tensor, pts_norm: torch.Tensor,
                  device: torch.device, num_queries: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init = torch.rand(1, num_queries, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method = nova_cfg.get("fm_sampling", "euler")
    T_grid = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def norm_to_world(pts_norm_np: np.ndarray, norm_factor: float, c2w: np.ndarray) -> np.ndarray:
    pts_cam = pts_norm_np / 3.0 * norm_factor
    ones = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


def chamfer(a: np.ndarray, b: np.ndarray, subsample: int = 10_000) -> float:
    rng = np.random.default_rng(0)
    if len(a) > subsample:
        a = a[rng.choice(len(a), subsample, replace=False)]
    if len(b) > subsample:
        b = b[rng.choice(len(b), subsample, replace=False)]
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    d = torch.cdist(a_t, b_t)
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    starts = list(args.starts)
    frame_stride = args.frame_stride

    # ── load checkpoint ───────────────────────────────────────────────────────
    print(f"[eval] Loading checkpoint from {args.checkpoint}")
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # ── scene data ────────────────────────────────────────────────────────────
    replica_root = args.replica_root
    room_dir = replica_root / args.room
    results_dir = room_dir / "results"
    mesh_ply = replica_root / f"{args.room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    h_nat, w_nat = int(cam["h"]), int(cam["w"])
    depth_scale = float(cam["scale"])
    k_native = torch.tensor([
        [cam["fx"], 0.0, cam["cx"]],
        [0.0, cam["fy"], cam["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    k_proc = k_native.clone()
    k_proc[0] *= args.image_width / w_nat
    k_proc[1] *= args.image_height / h_nat

    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]

    print(f"[eval] Loading mesh from {mesh_ply}")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    # ── models ────────────────────────────────────────────────────────────────
    print(f"[eval] Loading NOVA3R from {NOVA3R_CKPT}")
    nova_model, nova_cfg = load_nova3r_model(str(NOVA3R_CKPT), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    acfg = adapter_config_from_state_dict(ckpt_data["state_dict"])
    print(f"[eval] Adapter: {acfg}")
    adapter = DA3ToNOVA3RAlignment(**acfg, drop=0.0).to(device)
    adapter.load_state_dict(ckpt_data["state_dict"])
    adapter.eval()

    print(f"[eval] Loading DA3 ({args.da3_model}, layers {args.da3_layer})")
    da3_model = load_da3_model(args.da3_model, device)

    # ── per-window eval ───────────────────────────────────────────────────────
    print(f"\n[eval] {len(starts)} windows | stride {frame_stride} | "
          f"GT seed={GT_SEED} | ODE seed={args.seed}\n")
    print(f"{'window':<20}  {'CD (pred↔gt)':>14}")
    print("-" * 38)

    cds: list[float] = []

    def load_depth(fid: int) -> torch.Tensor:
        d = Image.open(results_dir / f"depth{fid:06d}.png")
        return torch.from_numpy(np.asarray(d, dtype=np.float32) / depth_scale)

    def load_rgb(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img = img.resize((args.image_width, args.image_height), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

    with torch.no_grad():
        for start in starts:
            frame_ids = [all_frame_ids[start + i * frame_stride] for i in range(N_FRAMES)]
            poses_c2w = torch.from_numpy(
                np.stack([poses_all[fid] for fid in frame_ids])).float()
            first_c2w = poses_c2w[0]

            # build visible-mesh pool
            pool_list: list[torch.Tensor] = []
            for i, fid in enumerate(frame_ids):
                vis = crop_visible_world_points(
                    points_world=mesh_pts,
                    camera_to_world=poses_c2w[i],
                    intrinsics=k_native,
                    depth=load_depth(fid),
                    depth_tolerance=args.depth_tolerance,
                )
                pool_list.append(world_to_first_camera(vis["points_world"], first_c2w))
            pool = torch.cat(pool_list, dim=0)

            # single-seed NOVA3R encode → ground-truth z_star
            pts = sample_pts(pool, K, seed=GT_SEED)
            gt_tokens, pts_norm, nf = encode_once(nova_model, norm_mode, pts, device)
            gt_zstar = gt_tokens.unsqueeze(0)  # (1, 768, 128)

            # DA3 tokens → adapter → predicted z_star
            images = torch.stack([load_rgb(fid) for fid in frame_ids])
            intrinsics = torch.from_numpy(
                np.tile(k_proc.numpy()[None], (N_FRAMES, 1, 1))).float()
            da3 = extract_da3_tokens(
                da3_model, images, poses_c2w, intrinsics,
                layer_idx=args.da3_layer if len(args.da3_layer) > 1 else args.da3_layer[0],
                max_tokens=args.da3_max_tokens,
                device=device,
            )
            pred_zstar = adapter(da3.to(device)).cpu()

            # decode both with the same pts_norm conditioning
            gt_norm = decode_tokens(nova_model, nova_cfg, gt_zstar, pts_norm, device,
                                    args.num_queries, args.seed)
            pred_norm = decode_tokens(nova_model, nova_cfg, pred_zstar, pts_norm, device,
                                      args.num_queries, args.seed)

            c2w = first_c2w.numpy()
            gt_world = norm_to_world(gt_norm, nf, c2w)
            pred_world = norm_to_world(pred_norm, nf, c2w)

            cd = chamfer(gt_world, pred_world, args.subsample)
            cds.append(cd)
            print(f"  start={start:<14}  {cd:>14.6f}")

    print("-" * 38)
    print(f"  {'mean':<20}  {np.mean(cds):>14.6f}")
    print(f"\n[result] mean Chamfer distance (pred vs. single-seed GT): {np.mean(cds):.6f} m")


if __name__ == "__main__":
    main()
