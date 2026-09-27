#!/usr/bin/env python3
"""
Cache stride-1 windows of 8 frames around a chosen validation window,
with consensus z_stars computed from k-means over N subsampling seeds.

Windows are named by their start frame: scripts/data/windows/start_{n:04d}/

Each window contains:
  da3_tokens.pt        (1, 2048, 2048)
  z_star_consensus.pt  (1, 768, 128)   k-means consensus over n_seeds subsamples
  pts_norm.pt          (1, 8192, 3)    seed-0 subsample for decode conditioning
  meta.pt

Usage:
  ! python scripts/cache_consecutive_windows.py --replica-root datasets/replica
  ! python scripts/cache_consecutive_windows.py --replica-root datasets/replica \\
        --val-start 24 --n-left 3 --n-right 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT   = Path(__file__).resolve().parents[1]
# nova3r_lib holds the active model code + the scene_ae checkpoint on this machine
# (the bare `nova3r/` copy has no checkpoint and differs in decoder/encoder code).
NOVA3R_ROOT = REPO_ROOT / os.environ.get("NOVA3R_DIR", "nova3r_lib")
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model              # noqa: E402
from nova3r.inference import normalize_input                         # noqa: E402
from vc3r.replica import crop_visible_world_points  # noqa: E402
from multi_scene_train import load_da3_model, extract_da3_tokens     # noqa: E402
from sklearn.cluster import KMeans

N_FRAMES   = 8
N_SEEDS    = 20
K          = 8192
N_CLUSTERS = 768


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--replica-root",    type=Path, default=None)
    p.add_argument("--room",            default="room0")
    p.add_argument("--rooms",           nargs="*", default=None,
                   help="Cache multiple scenes. If omitted, --room is used.")
    p.add_argument("--all-scenes",      action="store_true",
                   help="Cache every Replica scene with a matching *_mesh.ply, results/, and traj.txt.")
    p.add_argument("--val-start",       type=int,  default=24,
                   help="Start frame of the validation window")
    p.add_argument("--n-left",          type=int,  default=3,
                   help="Number of non-overlapping windows to the left of val")
    p.add_argument("--n-right",         type=int,  default=3,
                   help="Number of non-overlapping windows to the right of val")
    p.add_argument("--train-starts",     type=int,  nargs="*", default=None,
                   help="Explicit training window starts; overrides n-left/n-right layout")
    p.add_argument("--windows-per-room", type=int, default=None,
                   help="Sample this many distinct training window starts per scene. "
                        "Overrides val/n-left/n-right layout and does not append a validation window.")
    p.add_argument("--sample-seed",      type=int, default=0,
                   help="Seed for --windows-per-room start sampling")
    p.add_argument("--no-val-cache",      action="store_true",
                   help="Only cache training starts; do not append val-start to the cache job")
    p.add_argument("--require-non-overlap", action="store_true",
                   help="Fail if any cached windows contain overlapping frame ids")
    p.add_argument("--frame-stride",     type=int,  default=1,
                   help="Frame stride within each 8-frame window")
    p.add_argument("--out-root",         type=Path, default=None,
                   help="Output cache root. Defaults to scripts/data/windows for stride 1, windows_s<N> otherwise.")
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--da3-model",       default="depth-anything/DA3-LARGE-1.1")
    p.add_argument("--da3-layer",       type=int, default=3)
    p.add_argument("--da3-max-tokens",  type=int, default=2048)
    p.add_argument("--image-height",    type=int, default=392)
    p.add_argument("--image-width",     type=int, default=518)
    return p.parse_args()


def window_starts(val_start: int, n_left: int, n_right: int, frame_stride: int) -> dict[str, list[int]]:
    spacing = N_FRAMES * frame_stride
    left  = [val_start - (i + 1) * spacing for i in range(n_left)][::-1]
    right = [val_start + spacing + i * spacing for i in range(n_right)]
    return {"train": left + right, "val": [val_start]}


def sample_window_starts(max_start: int, n_windows: int, seed: int) -> list[int]:
    """Sample distinct valid window starts; frame ranges are allowed to intersect."""
    n_valid = max_start + 1
    if n_windows > n_valid:
        raise ValueError(f"Requested {n_windows} windows but only {n_valid} valid starts exist")
    rng = np.random.default_rng(seed)
    return sorted(int(s) for s in rng.choice(n_valid, size=n_windows, replace=False))


def discover_scenes(replica_root: Path) -> list[str]:
    scenes = []
    for mesh in sorted(replica_root.glob("*_mesh.ply")):
        room = mesh.name.removesuffix("_mesh.ply")
        room_dir = replica_root / room
        if (room_dir / "results").is_dir() and (room_dir / "traj.txt").exists():
            scenes.append(room)
    if not scenes:
        raise FileNotFoundError(f"No Replica scenes found under {replica_root}")
    return scenes


def world_to_first_camera(pts_world: torch.Tensor,
                           first_c2w: torch.Tensor) -> torch.Tensor:
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


def consensus_zstar(model, norm_mode: str, pool: torch.Tensor, device: torch.device):
    all_tokens: list[torch.Tensor] = []
    pts_norm_0, nf_0 = None, None
    for seed in range(N_SEEDS):
        pts = sample_pts(pool, K, seed=seed)
        z, pnorm, nf = encode_once(model, norm_mode, pts, device)
        all_tokens.append(z)
        if seed == 0:
            pts_norm_0, nf_0 = pnorm, nf
    # K-means over all N×768 tokens → 768 representative centroids
    flat = torch.stack(all_tokens).reshape(N_SEEDS * N_CLUSTERS, -1).numpy()
    km   = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=5, max_iter=300, verbose=0)
    km.fit(flat)
    centroids = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    return centroids.unsqueeze(0), pts_norm_0, nf_0


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    replica_root = (args.replica_root if args.replica_root is not None
                    else REPO_ROOT / "datasets" / "replica")

    if args.all_scenes:
        rooms = discover_scenes(replica_root)
    elif args.rooms:
        rooms = list(args.rooms)
    else:
        rooms = [args.room]

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    H_nat, W_nat    = int(cam["h"]), int(cam["w"])
    depth_scale     = float(cam["scale"])
    K_native        = torch.tensor([
        [cam["fx"], 0., cam["cx"]],
        [0., cam["fy"], cam["cy"]],
        [0., 0., 1.]], dtype=torch.float32)
    K_proc = K_native.clone()
    K_proc[0] *= args.image_width  / W_nat
    K_proc[1] *= args.image_height / H_nat

    frame_stride  = int(args.frame_stride)

    if args.out_root is not None:
        out_root = args.out_root
    elif frame_stride == 1:
        out_root = REPO_ROOT / "scripts" / "data" / "windows"
    else:
        out_root = REPO_ROOT / "scripts" / "data" / f"windows_s{frame_stride}"

    multi_room = len(rooms) > 1 or args.all_scenes or args.rooms is not None

    print("[cache] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    print("[cache] Loading DA3 …")
    da3_model = load_da3_model(args.da3_model, device)

    def cache_room(room: str, room_index: int) -> None:
        room_dir    = replica_root / room
        results_dir = room_dir / "results"
        mesh_ply    = replica_root / f"{room}_mesh.ply"
        if not results_dir.is_dir():
            raise FileNotFoundError(f"Missing results directory: {results_dir}")
        if not mesh_ply.exists():
            raise FileNotFoundError(f"Missing mesh: {mesh_ply}")

        poses_all     = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
        frame_files   = sorted(results_dir.glob("frame*.jpg"))
        all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
        max_start     = len(all_frame_ids) - 1 - (N_FRAMES - 1) * frame_stride

        if args.windows_per_room is not None:
            starts = {
                "train": sample_window_starts(
                    max_start=max_start,
                    n_windows=int(args.windows_per_room),
                    seed=int(args.sample_seed) + room_index,
                ),
                "val": [],
            }
        elif args.train_starts is None:
            starts = window_starts(args.val_start, args.n_left, args.n_right, frame_stride)
        else:
            starts = {"train": list(args.train_starts), "val": [args.val_start]}
        if args.no_val_cache:
            starts["val"] = []
        all_starts = starts["train"] + starts["val"]

        for s in all_starts:
            if s < 0 or s > max_start:
                raise ValueError(f"[{room}] Window start {s} is out of range [0, {max_start}]")

        frame_sets: dict[tuple[int, ...], int] = {}
        for s in all_starts:
            frame_ids = tuple(all_frame_ids[s + i * frame_stride] for i in range(N_FRAMES))
            if frame_ids in frame_sets:
                raise ValueError(
                    f"[{room}] Window start {s} has identical frames to start {frame_sets[frame_ids]}"
                )
            frame_sets[frame_ids] = s

        if args.require_non_overlap:
            seen_frames: dict[int, int] = {}
            for s in all_starts:
                for fid in [all_frame_ids[s + i * frame_stride] for i in range(N_FRAMES)]:
                    if fid in seen_frames:
                        raise ValueError(
                            f"[{room}] Window start {s} overlaps frame {fid} with start {seen_frames[fid]}"
                        )
                    seen_frames[fid] = s

        print(f"[cache] room={room}  val_start={args.val_start}  "
              f"n_left={args.n_left}  n_right={args.n_right}  frame_stride={frame_stride}")
        print(f"  train starts: {starts['train']}")
        print(f"  val start:    {starts['val']}")

        print(f"[cache] [{room}] Loading mesh …")
        mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
        mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
        mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

        room_out_root = out_root / room if multi_room else out_root

        def load_depth(fid: int) -> torch.Tensor:
            d = Image.open(results_dir / f"depth{fid:06d}.png")
            return torch.from_numpy(np.asarray(d, dtype=np.float32) / depth_scale)

        def load_rgb(fid: int) -> torch.Tensor:
            img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
            img = img.resize((args.image_width, args.image_height), Image.BILINEAR)
            return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

        for start in all_starts:
            win_dir = room_out_root / f"start_{start:04d}"
            needed  = ["da3_tokens.pt", "z_star_consensus.pt", "pts_norm.pt", "meta.pt"]
            if all((win_dir / f).exists() for f in needed):
                print(f"  [{room}] start={start:04d} already cached, skipping")
                continue
            win_dir.mkdir(parents=True, exist_ok=True)

            frame_ids = [all_frame_ids[start + i * frame_stride] for i in range(N_FRAMES)]
            poses_c2w = torch.from_numpy(
                np.stack([poses_all[fid] for fid in frame_ids])).float()
            first_c2w = poses_c2w[0]

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

            spread = float((poses_c2w[:, :3, 3].numpy().max(0)
                            - poses_c2w[:, :3, 3].numpy().min(0)).max())
            print(f"  [{room}] start={start:04d}  frames {frame_ids[0]}–{frame_ids[-1]}  "
                  f"pool={len(pool):,}  spread={spread:.3f}m", flush=True)

            print(f"    encoding {N_SEEDS} seeds + k-means …", flush=True)
            consensus, pts_norm, norm_factor = consensus_zstar(
                nova_model, norm_mode, pool, device)

            images     = torch.stack([load_rgb(fid) for fid in frame_ids])
            intrinsics = torch.from_numpy(
                np.tile(K_proc.numpy()[None], (N_FRAMES, 1, 1))).float()
            da3_tokens = extract_da3_tokens(
                da3_model, images, poses_c2w, intrinsics,
                layer_idx=args.da3_layer,
                max_tokens=args.da3_max_tokens,
                device=device,
            )

            torch.save(da3_tokens, win_dir / "da3_tokens.pt")
            torch.save(consensus,  win_dir / "z_star_consensus.pt")
            torch.save(pts_norm,   win_dir / "pts_norm.pt")
            torch.save({
                "frame_ids":   frame_ids,
                "poses_c2w":   poses_c2w,
                "norm_factor": norm_factor,
                "first_c2w":   first_c2w,
                "start_idx":   start,
                "room":        room,
                "stride":      frame_stride,
                "n_seeds":     N_SEEDS,
            }, win_dir / "meta.pt")
            print(f"    saved → {win_dir}  norm_factor={norm_factor:.3f}m")

    for room_index, room in enumerate(rooms):
        cache_room(room, room_index)

    print(f"\n[cache] Done → {out_root}")


if __name__ == "__main__":
    main()
