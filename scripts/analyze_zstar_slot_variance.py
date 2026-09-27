#!/usr/bin/env python3
"""Analyze per-slot z_star variance with online Hungarian alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import linear_sum_assignment

from cache_consecutive_windows import (
    K,
    N_SEEDS,
    N_FRAMES,
    REPO_ROOT,
    crop_visible_world_points,
    encode_once,
    load_nova3r_model,
    sample_pts,
    world_to_first_camera,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("window", type=Path, nargs="?", default=None,
                   help="Cached window directory, e.g. scripts/data/.../office4/start_0104")
    p.add_argument("--room", default=None,
                   help="Replica scene to analyze when no cached window directory is supplied")
    p.add_argument("--start", type=int, default=None,
                   help="Window start index when --room is used")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Frame stride within the 8-frame window when --room is used")
    p.add_argument("--replica-root", type=Path, default=REPO_ROOT / "datasets" / "replica")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def tensor_quantiles(x: torch.Tensor) -> dict[str, float]:
    q = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    vals = torch.quantile(x.float().reshape(-1), q)
    return {f"q{int(round(float(k) * 100)):02d}": float(v) for k, v in zip(q, vals)}


def load_window_pool(
    window: Path,
    replica_root: Path,
    depth_tolerance: float,
) -> tuple[torch.Tensor, dict]:
    meta = torch.load(window / "meta.pt", map_location="cpu", weights_only=False)
    room = str(meta["room"])
    frame_ids = [int(v) for v in meta["frame_ids"]]
    first_c2w = meta["first_c2w"].float()

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    depth_scale = float(cam["scale"])
    k_native = torch.tensor([
        [cam["fx"], 0.0, cam["cx"]],
        [0.0, cam["fy"], cam["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    room_dir = replica_root / room
    results_dir = room_dir / "results"
    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    mesh = trimesh.load(str(replica_root / f"{room}_mesh.ply"), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    pool_list: list[torch.Tensor] = []
    for fid in frame_ids:
        depth = Image.open(results_dir / f"depth{fid:06d}.png")
        depth = torch.from_numpy(np.asarray(depth, dtype=np.float32) / depth_scale)
        pose_c2w = torch.from_numpy(poses_all[fid]).float()
        visible = crop_visible_world_points(
            points_world=mesh_pts,
            camera_to_world=pose_c2w,
            intrinsics=k_native,
            depth=depth,
            depth_tolerance=depth_tolerance,
        )
        pool_list.append(world_to_first_camera(visible["points_world"], first_c2w))

    return torch.cat(pool_list, dim=0), meta


def load_room_window_pool(
    room: str,
    start: int,
    frame_stride: int,
    replica_root: Path,
    depth_tolerance: float,
) -> tuple[torch.Tensor, dict]:
    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    depth_scale = float(cam["scale"])
    k_native = torch.tensor([
        [cam["fx"], 0.0, cam["cx"]],
        [0.0, cam["fy"], cam["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    room_dir = replica_root / room
    results_dir = room_dir / "results"
    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
    max_start = len(all_frame_ids) - 1 - (N_FRAMES - 1) * frame_stride
    if start < 0 or start > max_start:
        raise ValueError(f"[{room}] Window start {start} is out of range [0, {max_start}]")
    frame_ids = [all_frame_ids[start + i * frame_stride] for i in range(N_FRAMES)]

    mesh = trimesh.load(str(replica_root / f"{room}_mesh.ply"), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    first_c2w = torch.from_numpy(poses_all[frame_ids[0]]).float()
    pool_list: list[torch.Tensor] = []
    for fid in frame_ids:
        depth = Image.open(results_dir / f"depth{fid:06d}.png")
        depth = torch.from_numpy(np.asarray(depth, dtype=np.float32) / depth_scale)
        pose_c2w = torch.from_numpy(poses_all[fid]).float()
        visible = crop_visible_world_points(
            points_world=mesh_pts,
            camera_to_world=pose_c2w,
            intrinsics=k_native,
            depth=depth,
            depth_tolerance=depth_tolerance,
        )
        pool_list.append(world_to_first_camera(visible["points_world"], first_c2w))

    meta = {
        "room": room,
        "start_idx": int(start),
        "frame_ids": frame_ids,
        "first_c2w": first_c2w,
        "stride": int(frame_stride),
    }
    return torch.cat(pool_list, dim=0), meta


def hungarian_match_to_anchor(sample: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cost = torch.cdist(anchor, sample).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    if not np.array_equal(row_ind, np.arange(anchor.shape[0])):
        raise RuntimeError("Unexpected non-monotonic row assignment")
    col = torch.from_numpy(col_ind).long()
    matched = sample[col]
    distances = (anchor - matched).norm(dim=-1)
    return matched, distances


def summarize_online_step(
    aligned: torch.Tensor,
    match_distances: torch.Tensor | None,
    old_anchor: torch.Tensor | None,
    new_anchor: torch.Tensor,
) -> dict:
    mean = aligned.mean(dim=0)
    centered = aligned - mean.unsqueeze(0)
    per_slot_mse = centered.square().mean(dim=(0, 2))
    per_slot_l2_std = centered.norm(dim=-1).std(dim=0, unbiased=False)
    out = {
        "slot_centered_mse": tensor_quantiles(per_slot_mse),
        "slot_centered_l2_std": tensor_quantiles(per_slot_l2_std),
        "mean_slot_mse_mean": float(per_slot_mse.mean()),
        "mean_slot_l2_std_mean": float(per_slot_l2_std.mean()),
    }
    if match_distances is not None:
        out["new_sample_match_distance_l2"] = tensor_quantiles(match_distances)
        out["new_sample_match_distance_l2_mean"] = float(match_distances.mean())
    if old_anchor is not None:
        mean_shift = (new_anchor - old_anchor).norm(dim=-1)
        out["mean_shift_l2"] = tensor_quantiles(mean_shift)
        out["mean_shift_l2_mean"] = float(mean_shift.mean())
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.window is None and (args.room is None or args.start is None):
        raise SystemExit("Supply either a cached window path or both --room and --start.")

    run_name = (
        f"{args.room}_start{args.start:04d}_s{args.frame_stride}_runs{args.n_seeds}"
        if args.window is None
        else f"{args.window.parent.name}_{args.window.name}_runs{args.n_seeds}"
    )
    out_dir = args.out_dir or (
        REPO_ROOT / "outputs" / "zstar_slot_variance" / run_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.window is None:
        print(f"[load] room={args.room} start={args.start} frame_stride={args.frame_stride}")
        pool, meta = load_room_window_pool(
            args.room, args.start, args.frame_stride, args.replica_root, args.depth_tolerance)
    else:
        print(f"[load] window={args.window}")
        pool, meta = load_window_pool(args.window, args.replica_root, args.depth_tolerance)
    print(f"[load] room={meta['room']} start={meta['start_idx']} pool={len(pool):,}")

    print("[model] Loading NOVA3R")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    print(f"[online] {args.n_seeds} NOVA3R runs of {K} sampled points")
    raw_samples = []
    aligned_samples = []
    means_by_run = []
    online_steps = []

    for seed in range(args.n_seeds):
        pts = sample_pts(pool, K, seed=seed)
        z, _, _ = encode_once(nova_model, norm_mode, pts, device)
        raw_samples.append(z)

        if seed == 0:
            matched = z
            match_distances = None
            old_anchor = None
            print(
                f"  run={seed:02d} initial_anchor token_l2_mean={z.norm(dim=-1).mean():.4f}",
                flush=True,
            )
        else:
            old_anchor = means_by_run[-1]
            matched, match_distances = hungarian_match_to_anchor(z, old_anchor)
            print(
                f"  run={seed:02d} token_l2_mean={z.norm(dim=-1).mean():.4f} "
                f"match_l2_median={torch.median(match_distances):.4f}",
                flush=True,
            )

        aligned_samples.append(matched)
        aligned_t = torch.stack(aligned_samples)
        new_anchor = aligned_t.mean(dim=0)
        means_by_run.append(new_anchor)

        step_summary = summarize_online_step(
            aligned=aligned_t,
            match_distances=match_distances,
            old_anchor=old_anchor,
            new_anchor=new_anchor,
        )
        step_summary["run"] = int(seed)
        step_summary["n_aligned_samples"] = int(seed + 1)
        online_steps.append(step_summary)

    raw_t = torch.stack(raw_samples)
    aligned_t = torch.stack(aligned_samples)
    means_t = torch.stack(means_by_run)
    torch.save(raw_t, out_dir / "raw_tokens.pt")
    torch.save(aligned_t, out_dir / "aligned_tokens.pt")
    torch.save(means_t, out_dir / "means_by_run.pt")
    torch.save(means_t[-1], out_dir / "mean_final.pt")

    summary = {
        "window": str(args.window) if args.window is not None else None,
        "room": str(meta["room"]),
        "start_idx": int(meta["start_idx"]),
        "frame_ids": [int(v) for v in meta["frame_ids"]],
        "frame_stride": int(meta.get("stride", args.frame_stride)),
        "n_seeds": int(args.n_seeds),
        "process": "online_hungarian_to_running_mean",
        "raw_token_l2": tensor_quantiles(raw_t.norm(dim=-1)),
        "online_steps": online_steps,
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
