#!/usr/bin/env python3
"""
Cache N evenly-spaced stride-150 windows from room0, including DA3 tokens.

Saves to experiments/overfit_8frames/data/multi_N{n}_s150/sample_*/

Usage:
  python experiments/overfit_8frames/cache_stride150.py
  python experiments/overfit_8frames/cache_stride150.py --n-windows 10 --replica-root datasets/replica
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

REPO_ROOT   = Path(__file__).resolve().parents[2]
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.inference import normalize_input                     # noqa: E402
from multi_scene_train import (                                  # noqa: E402
    load_da3_model, extract_da3_tokens,
    crop_frustum_world_points, world_to_first_camera,
    sample_points, encode_nova3r,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",       type=Path,
                   default=Path(__file__).parent / "config.yaml")
    p.add_argument("--replica-root", type=Path, default=None)
    p.add_argument("--room",         default="room0")
    p.add_argument("--stride",       type=int, default=150)
    p.add_argument("--n-windows",    type=int, default=10)
    p.add_argument("--device",       default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    device = torch.device(args.device)

    replica_root = (args.replica_root if args.replica_root is not None
                    else Path(str(cfg.replica_root)))
    room     = args.room
    stride   = args.stride
    n_frames = int(cfg.num_frames)

    out_root = (REPO_ROOT / "experiments" / "overfit_8frames" / "data"
                / f"multi_N{args.n_windows}_s{stride}")

    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                      [0., cam_p["fy"], cam_p["cy"]],
                      [0., 0., 1.]], dtype=np.float32)
    H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
    K_proc = K_nat.copy(); K_proc[0] *= W_proc / W_nat; K_proc[1] *= H_proc / H_nat
    K_nat_t = torch.from_numpy(K_nat)

    room_dir      = replica_root / room
    results_dir   = room_dir / "results"
    mesh_ply      = replica_root / f"{room}_mesh.ply"
    poses_all     = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files   = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]

    # Evenly-spaced start indices across valid range (exclusive of 0 = already cached)
    max_start = len(all_frame_ids) - 1 - (n_frames - 1) * stride
    start_indices = [int(x) for x in
                     np.round(np.linspace(0, max_start, args.n_windows + 2)).astype(int)[1:-1]]
    print(f"[cache] room={room}  stride={stride}  n={args.n_windows}")
    print(f"[cache] start indices: {start_indices}")
    print(f"[cache] output: {out_root}")

    print("[cache] Loading mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    print("[cache] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    print("[cache] Loading DA3 …")
    da3_model = load_da3_model(str(cfg.da3_model), device)

    def load_rgb(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img = img.resize((W_proc, H_proc), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

    for si, start_idx in enumerate(start_indices):
        sample_dir = out_root / f"sample_{si:03d}"
        if all((sample_dir / f).exists()
               for f in ["da3_tokens.pt", "z_star.pt", "pts_norm.pt", "meta.pt"]):
            print(f"  sample {si:03d} already cached, skipping")
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)

        frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
        poses_c2w = torch.from_numpy(
            np.stack([poses_all[fid] for fid in frame_ids])).float()
        first_c2w = poses_c2w[0]

        pts_cam_list = []
        for i in range(n_frames):
            frust = crop_frustum_world_points(
                mesh_pts, poses_c2w[i], K_nat_t, (H_nat, W_nat))
            pts_cam_list.append(world_to_first_camera(frust, first_c2w))
        tok_pts = sample_points(
            torch.cat(pts_cam_list, dim=0),
            int(cfg.mesh_sample_points), seed=start_idx)

        z_star, pts_norm = encode_nova3r(
            nova_model, nova_cfg, tok_pts.unsqueeze(0), device, seed=start_idx)
        norm_factor = float(tok_pts.norm(dim=-1).median().clamp(0.01, 100.0))

        images     = torch.stack([load_rgb(fid) for fid in frame_ids])
        intrinsics = torch.from_numpy(
            np.tile(K_proc[None], (n_frames, 1, 1))).float()
        da3_tokens = extract_da3_tokens(
            da3_model, images, poses_c2w, intrinsics,
            layer_idx=int(cfg.da3_source_layer_index),
            max_tokens=int(cfg.da3_max_source_tokens),
            device=device,
        )

        torch.save(da3_tokens, sample_dir / "da3_tokens.pt")
        torch.save(z_star,     sample_dir / "z_star.pt")
        torch.save(pts_norm,   sample_dir / "pts_norm.pt")
        torch.save({
            "frame_ids":   frame_ids,
            "poses_c2w":   poses_c2w,
            "norm_factor": norm_factor,
            "first_c2w":   first_c2w,
            "start_idx":   start_idx,
            "room":        room,
            "stride":      stride,
        }, sample_dir / "meta.pt")

        cam_pos = poses_c2w[:, :3, 3].numpy()
        spread  = float((cam_pos.max(0) - cam_pos.min(0)).max())
        print(f"  sample {si:03d}  start={start_idx}  frames {frame_ids[0]}-{frame_ids[-1]}  "
              f"norm_factor={norm_factor:.3f}m  spread={spread:.2f}m", flush=True)

    print(f"[cache] Done. {args.n_windows} windows saved to {out_root}")


if __name__ == "__main__":
    main()
