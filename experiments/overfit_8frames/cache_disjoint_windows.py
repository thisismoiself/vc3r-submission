#!/usr/bin/env python3
"""Cache room windows, optionally excluding frames from one validation window."""
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "overfit_8frames"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from multi_scene_train import (  # noqa: E402
    crop_frustum_world_points,
    encode_nova3r,
    extract_da3_tokens,
    load_da3_model,
    load_nova3r_model,
    sample_points,
    world_to_first_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--room", default="room0")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--val-start-idx", type=int, default=929)
    parser.add_argument("--ignore-overlap", action="store_true",
                        help="Cache every valid window; do not exclude validation-frame overlap")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete_sample(sample_dir: Path) -> bool:
    return all((sample_dir / name).exists() for name in [
        "da3_tokens.pt", "z_star.pt", "pts_norm.pt", "meta.pt"
    ])


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    n_frames = int(cfg.num_frames)
    stride = int(args.stride)
    room = args.room

    replica_root = Path(str(cfg.replica_root))
    room_dir = replica_root / room
    results_dir = room_dir / "results"
    mesh_ply = replica_root / f"{room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                      [0., cam_p["fy"], cam_p["cy"]],
                      [0., 0., 1.]], dtype=np.float32)
    H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
    K_proc = K_nat.copy()
    K_proc[0] *= W_proc / W_nat
    K_proc[1] *= H_proc / H_nat
    K_nat_t = torch.from_numpy(K_nat)

    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
    max_start_idx = len(all_frame_ids) - 1 - (n_frames - 1) * stride

    val_frames = set()
    if not args.ignore_overlap:
        val_frames = {
            all_frame_ids[args.val_start_idx + i * stride]
            for i in range(n_frames)
        }
    train_starts = []
    for start_idx in range(max_start_idx + 1):
        frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
        if args.ignore_overlap or val_frames.isdisjoint(frame_ids):
            train_starts.append(start_idx)

    data_root = EXP_DIR / "data" / f"multi_scene_N{len(train_starts)}_s{stride}" / room
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[cache_disjoint] room={room}")
    if args.ignore_overlap:
        print("[cache_disjoint] overlap exclusion disabled; caching every valid window")
    else:
        print(f"[cache_disjoint] val_start={args.val_start_idx} val_frames={sorted(val_frames)}")
    print(f"[cache_disjoint] train_windows={len(train_starts)} data_root={data_root}")
    print(f"[cache_disjoint] device={device}")

    needs = [
        (si, start_idx)
        for si, start_idx in enumerate(train_starts)
        if args.force or not complete_sample(data_root / f"sample_{si:03d}")
    ]
    print(f"[cache_disjoint] missing_or_forced={len(needs)} / {len(train_starts)}")
    if not needs:
        return

    print("[cache_disjoint] Loading NOVA3R ...")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    print("[cache_disjoint] Loading DA3 ...")
    da3_model = load_da3_model(str(cfg.da3_model), device)

    print("[cache_disjoint] Loading mesh ...")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    def load_rgb(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img = img.resize((W_proc, H_proc), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

    for progress_idx, (si, start_idx) in enumerate(needs, start=1):
        sample_dir = data_root / f"sample_{si:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
        poses_c2w = torch.from_numpy(np.stack([poses_all[fid] for fid in frame_ids])).float()

        print(
            f"[cache_disjoint] {progress_idx:04d}/{len(needs):04d} "
            f"sample_{si:03d} start={start_idx} frames={frame_ids}",
            flush=True,
        )

        images = torch.stack([load_rgb(fid) for fid in frame_ids])
        intrinsics = torch.from_numpy(np.tile(K_proc[None], (n_frames, 1, 1))).float()

        first_c2w = poses_c2w[0]
        pts_cam_list = []
        for i in range(n_frames):
            frust = crop_frustum_world_points(mesh_pts, poses_c2w[i], K_nat_t, (H_nat, W_nat))
            pts_cam_list.append(world_to_first_camera(frust, first_c2w))

        all_frust = torch.cat(pts_cam_list, dim=0)
        tok_pts = sample_points(all_frust, int(cfg.mesh_sample_points), seed=start_idx)
        z_star, pts_norm = encode_nova3r(
            nova_model, nova_cfg, tok_pts.unsqueeze(0), device, seed=start_idx * 1009
        )
        da3_tokens = extract_da3_tokens(
            da3_model, images, poses_c2w, intrinsics,
            layer_idx=int(cfg.da3_source_layer_index),
            max_tokens=int(cfg.da3_max_source_tokens),
            device=device,
        )

        torch.save(da3_tokens, sample_dir / "da3_tokens.pt")
        torch.save(z_star, sample_dir / "z_star.pt")
        torch.save(pts_norm, sample_dir / "pts_norm.pt")
        torch.save({
            "frame_ids": frame_ids,
            "poses_c2w": poses_c2w,
            "start_idx": start_idx,
            "stride": stride,
            "room": room,
            "val_start_idx": args.val_start_idx,
            "val_frame_ids": sorted(val_frames),
            "split": "all_windows" if args.ignore_overlap else "train_disjoint_from_val",
        }, sample_dir / "meta.pt")

    print("[cache_disjoint] Done.")


if __name__ == "__main__":
    main()
