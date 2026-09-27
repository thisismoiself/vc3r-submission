#!/usr/bin/env python3
"""Cache windows with online-Hungarian z_star distribution targets.

For each window, NOVA3R is run on multiple point samples; each run is
Hungarian-matched to the running slot mean to build mean/variance targets.

Each output window contains:
  da3_tokens.pt, z_star_online_{mean,var,var_eff,samples}.pt, pts_norm.pt, meta.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import linear_sum_assignment

from cache_consecutive_windows import (
    K, N_CLUSTERS, REPO_ROOT,
    crop_visible_world_points, discover_scenes, encode_once,
    extract_da3_tokens, load_da3_model, load_nova3r_model,
    sample_pts, world_to_first_camera,
)

EXPECTED_FILES = [
    "da3_tokens.pt", "z_star_online_mean.pt", "z_star_online_var.pt",
    "z_star_online_var_eff.pt", "z_star_online_samples.pt", "pts_norm.pt", "meta.pt",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--replica-root", type=Path, default=None)
    p.add_argument("--room", default="room0")
    p.add_argument("--rooms", nargs="*", default=None, help="Cache multiple scenes.")
    p.add_argument("--all-scenes", action="store_true", help="Cache every Replica scene.")
    p.add_argument("--windows-per-room", type=int, default=None,
                   help="Number of windows to sample per scene.")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--span-min", type=int, default=20,
                   help="Log-uniform span lower bound (frames between first and last selected).")
    p.add_argument("--span-max", type=int, default=500,
                   help="Log-uniform span upper bound.")
    p.add_argument("--n-frames-min", type=int, default=8,
                   help="Log-uniform frame count lower bound.")
    p.add_argument("--n-frames-max", type=int, default=200,
                   help="Log-uniform frame count upper bound (also capped at span // 2).")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Output root (default: scripts/data/online_hungarian_zstar_windows).")
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--var-floor-quantile", type=float, default=0.05)
    p.add_argument("--save-raw-tokens", action="store_true")
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--da3-model", default="depth-anything/DA3-LARGE-1.1")
    p.add_argument("--da3-layer", type=int, nargs="+", default=[3])
    p.add_argument("--da3-max-tokens", type=int, default=2048)
    p.add_argument("--image-height", type=int, default=392)
    p.add_argument("--image-width", type=int, default=518)
    p.add_argument("--flip", action="store_true", help="50%% horizontal-flip augmentation.")
    return p.parse_args()


def hungarian_match_to_anchor(
    sample: torch.Tensor, anchor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cost = torch.cdist(anchor, sample).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    if not np.array_equal(row_ind, np.arange(anchor.shape[0])):
        raise RuntimeError("Unexpected non-monotonic row assignment")
    col = torch.from_numpy(col_ind).long()
    matched = sample[col]
    return matched, (anchor - matched).norm(dim=-1)


def stratified_frame_sample(
    start: int, end: int, n_frames: int, rng: np.random.Generator,
) -> list[int]:
    """Sample n_frames distinct positions from [start, end], sorted.

    Draws without replacement then sorts — guaranteed no duplicates and
    uniform coverage of the span in expectation.
    """
    return sorted(int(x) for x in rng.choice(end - start + 1, size=n_frames, replace=False) + start)


@torch.no_grad()
def online_hungarian_zstar(model, norm_mode, pool, device, n_seeds, var_floor_quantile):
    if n_seeds < 2:
        raise ValueError("--n-seeds must be at least 2")
    raw_tokens, aligned_tokens, summaries = [], [], []
    pts_norm_0 = nf_0 = None
    for seed in range(n_seeds):
        pts = sample_pts(pool, K, seed=seed)
        z, pts_norm, nf = encode_once(model, norm_mode, pts, device)
        raw_tokens.append(z)
        if seed == 0:
            matched, dist = z, None
            pts_norm_0, nf_0 = pts_norm, nf
            print(f"      run=00 anchor l2={z.norm(dim=-1).mean():.4f}", flush=True)
        else:
            matched, dist = hungarian_match_to_anchor(z, torch.stack(aligned_tokens).mean(0))
            print(f"      run={seed:02d} l2={z.norm(dim=-1).mean():.4f} "
                  f"match_median={torch.median(dist):.4f}", flush=True)
        aligned_tokens.append(matched)
        summaries.append({
            "run": seed,
            "match_l2_mean":   float(dist.mean())                if dist is not None else None,
            "match_l2_median": float(torch.median(dist))         if dist is not None else None,
            "match_l2_q95":    float(torch.quantile(dist, 0.95)) if dist is not None else None,
        })

    aligned = torch.stack(aligned_tokens)
    mean = aligned.mean(0)
    var = aligned.var(0, unbiased=False)
    var_floor = torch.quantile(var.reshape(-1), var_floor_quantile)
    return {
        "mean": mean.unsqueeze(0),
        "var": var.unsqueeze(0),
        "var_eff": torch.clamp(var, min=float(var_floor)).unsqueeze(0),
        "aligned": aligned,
        "raw": torch.stack(raw_tokens),
        "pts_norm": pts_norm_0,
        "norm_factor": nf_0,
        "var_floor": float(var_floor),
        "match_summaries": summaries,
    }


def ensure_safe_to_write(win_dir: Path, expected: list[str]) -> bool:
    existing = [n for n in expected if (win_dir / n).exists()]
    if len(existing) == len(expected):
        return False
    if existing:
        raise FileExistsError(f"Refusing partial write under {win_dir}; existing: {existing}")
    if win_dir.exists() and any(win_dir.iterdir()):
        raise FileExistsError(f"Non-empty directory: {win_dir}")
    return True


def main() -> None:
    args = parse_args()
    if args.windows_per_room is None:
        raise ValueError("--windows-per-room is required")
    if args.span_min >= args.span_max:
        raise ValueError("--span-min must be strictly less than --span-max")
    if args.n_frames_min >= args.n_frames_max:
        raise ValueError("--n-frames-min must be strictly less than --n-frames-max")

    device = torch.device(args.device)
    replica_root = args.replica_root or REPO_ROOT / "datasets" / "replica"

    if args.all_scenes:
        rooms = discover_scenes(replica_root)
    elif args.rooms:
        rooms = list(args.rooms)
    else:
        rooms = [args.room]

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

    # c2w @ X_FLIP negates camera x, giving the correct pose for a horizontally flipped image.
    X_FLIP = torch.diag(torch.tensor([-1., 1., 1., 1.]))
    out_root = args.out_root or REPO_ROOT / "scripts" / "data" / "online_hungarian_zstar_windows"
    multi_room = len(rooms) > 1 or args.all_scenes or args.rooms is not None

    print("[cache-online] Loading NOVA3R")
    ckpt = str(REPO_ROOT / os.environ.get("NOVA3R_DIR", "nova3r_lib") / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    print("[cache-online] Loading DA3")
    da3_model = load_da3_model(args.da3_model, device)

    def cache_room(room: str, room_index: int) -> None:
        room_dir = replica_root / room
        results_dir = room_dir / "results"
        mesh_ply = replica_root / f"{room}_mesh.ply"
        if not results_dir.is_dir():
            raise FileNotFoundError(f"Missing results directory: {results_dir}")
        if not mesh_ply.exists():
            raise FileNotFoundError(f"Missing mesh: {mesh_ply}")

        poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
        all_frame_ids = [int(p.stem.replace("frame", ""))
                         for p in sorted(results_dir.glob("frame*.jpg"))]
        n_frames_total = len(all_frame_ids)

        # Arcsine (Beta(0.5,0.5)) start: pushes long-span windows toward sequence edges,
        # counteracting the bias where middle frames are reachable from more start positions.
        n = int(args.windows_per_room)
        span_rng  = np.random.default_rng(int(args.sample_seed) + room_index * 100000 + 5_000_000)
        nf_rng    = np.random.default_rng(int(args.sample_seed) + room_index * 100000 + 6_000_000)
        start_rng = np.random.default_rng(int(args.sample_seed) + room_index * 100000 + 7_000_000)
        frame_rng = np.random.default_rng(int(args.sample_seed) + room_index * 100000 + 8_000_000)

        all_windows: list[tuple[list[int], int]] = []  # (frame_ids, span)
        seen_window_frames: set[tuple[int, ...]] = set()
        attempts = 0
        max_attempts = max(1000, n * 100)

        while len(all_windows) < n and attempts < max_attempts:
            attempts += 1
            span = int(np.clip(
                round(math.exp(span_rng.uniform(math.log(args.span_min), math.log(args.span_max)))),
                args.span_min, args.span_max,
            ))
            n_frames_upper = min(args.n_frames_max, span // 2)
            if n_frames_upper < args.n_frames_min:
                continue
            n_frames = int(np.clip(
                round(math.exp(nf_rng.uniform(math.log(args.n_frames_min), math.log(n_frames_upper)))),
                args.n_frames_min, n_frames_upper,
            ))
            slack = n_frames_total - 1 - span
            if slack < 0:
                continue
            start_pos = int(round(start_rng.beta(0.5, 0.5) * slack))
            end_pos = start_pos + span

            positions = stratified_frame_sample(start_pos, end_pos, n_frames, frame_rng)
            fids = tuple(all_frame_ids[p] for p in positions)
            if fids in seen_window_frames:
                continue
            seen_window_frames.add(fids)
            all_windows.append((list(fids), span))

        if len(all_windows) < n:
            raise ValueError(
                f"[{room}] could only sample {len(all_windows)} unique windows "
                f"out of requested {n} after {attempts} attempts"
            )

        print(f"[cache-online] room={room}  span=[{args.span_min},{args.span_max}]  "
              f"n_frames=[{args.n_frames_min},{args.n_frames_max}]  windows={n}")

        mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
        mesh_pts = torch.from_numpy(
            trimesh.sample.sample_surface(mesh, 2_000_000)[0].astype(np.float32)
        )
        room_out_root = out_root / room if multi_room else out_root

        def load_depth(fid: int) -> torch.Tensor:
            return torch.from_numpy(
                np.asarray(Image.open(results_dir / f"depth{fid:06d}.png"), np.float32)
                / depth_scale
            )

        def load_rgb(fid: int) -> torch.Tensor:
            img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
            img = img.resize((args.image_width, args.image_height), Image.BILINEAR)
            return torch.from_numpy(np.array(img, np.float32)).permute(2, 0, 1) / 255.0

        expected = EXPECTED_FILES + (["z_star_online_raw.pt"] if args.save_raw_tokens else [])
        for win_idx, (frame_ids, span) in enumerate(all_windows):
            n_frames = len(frame_ids)
            win_dir = room_out_root / f"{frame_ids[0]}_{frame_ids[-1]}_{n_frames}f_{span}s"
            if not ensure_safe_to_write(win_dir, expected):
                print(f"  [{room}] {frame_ids[0]}_{frame_ids[-1]}_{n_frames}f_{span}s already cached, skipping")
                continue
            win_dir.mkdir(parents=True, exist_ok=True)
            poses_c2w = torch.from_numpy(np.stack([poses_all[fid] for fid in frame_ids])).float()
            first_c2w = poses_c2w[0]
            spread = float((poses_c2w[:, :3, 3].numpy().max(0)
                            - poses_c2w[:, :3, 3].numpy().min(0)).max())

            apply_flip = False
            if args.flip:
                flip_rng = np.random.default_rng(
                    int(args.sample_seed) + room_index * 100000 + win_idx)
                apply_flip = bool(flip_rng.integers(0, 2))

            pool_list = []
            for i, fid in enumerate(frame_ids):
                vis = crop_visible_world_points(
                    points_world=mesh_pts, camera_to_world=poses_c2w[i],
                    intrinsics=k_native, depth=load_depth(fid), depth_tolerance=args.depth_tolerance,
                )
                pool_list.append(world_to_first_camera(vis["points_world"], first_c2w))
            pool = torch.cat(pool_list, dim=0)
            pool_encode = pool.clone()
            if apply_flip:
                pool_encode[:, 0] = -pool_encode[:, 0]  # flip negates pool x

            aug_label = " [flip]" if apply_flip else ""
            print(f"  [{room}] win={win_idx:04d} span={span} n_frames={n_frames} "
                  f"frames={frame_ids[0]}-{frame_ids[-1]} "
                  f"pool={len(pool):,} spread={spread:.3f}m{aug_label}", flush=True)

            print(f"    encoding {args.n_seeds} online-Hungarian runs")
            zstats = online_hungarian_zstar(
                nova_model, norm_mode, pool_encode, device,
                n_seeds=int(args.n_seeds), var_floor_quantile=float(args.var_floor_quantile),
            )

            images = torch.stack([load_rgb(fid) for fid in frame_ids])
            if apply_flip:
                images = torch.flip(images, dims=[-1])
            poses_da3 = poses_c2w @ X_FLIP if apply_flip else poses_c2w

            intrinsics = torch.from_numpy(np.tile(k_proc.numpy()[None], (n_frames, 1, 1))).float()
            da3_tokens = extract_da3_tokens(
                da3_model, images, poses_da3, intrinsics,
                layer_idx=args.da3_layer, max_tokens=args.da3_max_tokens, device=device,
            )

            torch.save(da3_tokens,          win_dir / "da3_tokens.pt")
            torch.save(zstats["mean"],      win_dir / "z_star_online_mean.pt")
            torch.save(zstats["var"],       win_dir / "z_star_online_var.pt")
            torch.save(zstats["var_eff"],   win_dir / "z_star_online_var_eff.pt")
            torch.save(zstats["aligned"],   win_dir / "z_star_online_samples.pt")
            torch.save(zstats["pts_norm"],  win_dir / "pts_norm.pt")
            if args.save_raw_tokens:
                torch.save(zstats["raw"],   win_dir / "z_star_online_raw.pt")
            torch.save({
                "frame_ids": frame_ids, "poses_c2w": poses_c2w,
                "norm_factor": zstats["norm_factor"], "first_c2w": first_c2w,
                "span": span, "room": room,
                "target_mode": "online_hungarian_running_mean",
                "n_seeds": int(args.n_seeds), "target_tokens": N_CLUSTERS, "target_dim": 128,
                "var_floor_quantile": float(args.var_floor_quantile),
                "var_floor": float(zstats["var_floor"]),
                "match_summaries": zstats["match_summaries"],
                "flip": apply_flip,
            }, win_dir / "meta.pt")
            print(f"    saved -> {win_dir} var_floor={zstats['var_floor']:.6g}{aug_label}")

    for room_index, room in enumerate(rooms):
        cache_room(room, room_index)

    print(f"\n[cache-online] Done -> {out_root}")


if __name__ == "__main__":
    main()
