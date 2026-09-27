#!/usr/bin/env python3
"""Decode online-variance Hungarian targets/predictions into a Rerun file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

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
from train_hungarian import decode_tokens, norm_to_world  # noqa: E402

CFG_PATH = OVERFIT_SRC / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rrd-out", type=Path, required=True)
    p.add_argument("--num-queries", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shade-axis", choices=["x", "y", "z"], default="y")
    p.add_argument("--rot90-axis", choices=["none", "x", "y", "z", "-x", "-y", "-z"],
                   default="x")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_window(root: Path, start: int):
    win_dir = root / f"start_{start:04d}"
    meta = torch.load(win_dir / "meta.pt", map_location="cpu", weights_only=False)
    da3 = torch.load(win_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    mean = torch.load(win_dir / "z_star_online_mean.pt", map_location="cpu", weights_only=True)
    pts_norm = torch.load(win_dir / "pts_norm.pt", map_location="cpu", weights_only=True)
    return meta, da3, mean, pts_norm


def load_window_dir(win_dir: Path):
    meta = torch.load(win_dir / "meta.pt", map_location="cpu", weights_only=False)
    da3 = torch.load(win_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    mean = torch.load(win_dir / "z_star_online_mean.pt", map_location="cpu", weights_only=True)
    pts_norm = torch.load(win_dir / "pts_norm.pt", map_location="cpu", weights_only=True)
    return meta, da3, mean, pts_norm


def val_window_items(val_configs: list[dict]) -> list[tuple[Path, Path, int]]:
    items = []
    for cfg_item in val_configs:
        root = Path(cfg_item["root"])
        for start_item in cfg_item["starts"]:
            if isinstance(start_item, (str, Path)) and str(start_item).startswith(("courses/", "/", ".")):
                win_dir = Path(start_item)
                start = int(win_dir.name.split("_")[1])
            else:
                start = int(start_item)
                win_dir = root / f"start_{start:04d}"
            items.append((root, win_dir, start))
    return items


def rotate_points(pts: np.ndarray, axis: str) -> np.ndarray:
    if axis == "none":
        return pts
    rotations = {
        "x": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        "-x": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32),
        "y": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
        "-y": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        "z": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        "-z": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
    }
    return pts @ rotations[axis].T


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = ckpt_data.get("metadata", {})

    cfg = OmegaConf.load(CFG_PATH)
    OmegaConf.set_struct(cfg, False)
    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim),
        hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens),
        target_dim=int(cfg.target_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        drop=0.0,
    ).to(device)
    adapter.load_state_dict(ckpt_data["state_dict"])
    adapter.eval()

    print("[decode] Loading NOVA3R")
    nova_ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(nova_ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    val_items = val_window_items(metadata["val_configs"])

    with torch.no_grad():
        loaded = []
        for root, win_dir, start in val_items:
            meta, da3, mean, pts_norm = load_window_dir(win_dir)
            pred = adapter(da3.to(device)).cpu()
            loaded.append((root, start, meta, mean, pred, pts_norm))

    import rerun as rr

    args.rrd_out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("online_var_hungarian_validation", spawn=False)
    rr.save(str(args.rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    blue = np.array([35, 115, 255], dtype=np.uint8)
    red = np.array([230, 45, 55], dtype=np.uint8)
    green = np.array([45, 205, 90], dtype=np.uint8)
    purple = np.array([165, 85, 245], dtype=np.uint8)
    axis_index = {"x": 0, "y": 1, "z": 2}[args.shade_axis]

    def sub(pts: np.ndarray, n: int = 100_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    def shaded(pts: np.ndarray, base: np.ndarray, lo: float, hi: float) -> np.ndarray:
        t = np.clip((pts[:, axis_index] - lo) / (hi - lo + 1e-8), 0, 1)
        shade = 0.42 + 0.58 * t
        return np.clip(base[None, :].astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)

    for i, (root, start, meta, mean, pred, pts_norm) in enumerate(loaded):
        rr.set_time("val_window", sequence=i)
        nf = float(meta["norm_factor"])
        c2w = meta["poses_c2w"][0].numpy()

        print(f"[decode] val start={start} target mean")
        target_norm = decode_tokens(nova_model, nova_cfg, mean, pts_norm, device, args.num_queries, args.seed)
        print(f"[decode] val start={start} adapter pred")
        pred_norm = decode_tokens(nova_model, nova_cfg, pred, pts_norm, device, args.num_queries, args.seed)

        target_world = norm_to_world(target_norm, nf, c2w)
        pred_world = norm_to_world(pred_norm, nf, c2w)
        input_cam = pts_norm[0].numpy() / 3.0 * nf
        input_world = (c2w @ np.hstack([input_cam, np.ones((len(input_cam), 1), dtype=np.float32)]).T).T[:, :3]
        cameras = meta["poses_c2w"][:, :3, 3].numpy()

        print(f"[decode] start={start} target centroid={target_world.mean(0).round(3)} "
              f"pred centroid={pred_world.mean(0).round(3)}")

        target_world = rotate_points(target_world, args.rot90_axis)
        pred_world = rotate_points(pred_world, args.rot90_axis)
        input_world = rotate_points(input_world, args.rot90_axis)
        cameras = rotate_points(cameras, args.rot90_axis)
        all_axis = np.concatenate([
            target_world[:, axis_index],
            pred_world[:, axis_index],
            input_world[:, axis_index],
            cameras[:, axis_index],
        ])
        lo, hi = float(all_axis.min()), float(all_axis.max())

        target_sub = sub(target_world)
        pred_sub = sub(pred_world)
        input_sub = sub(input_world, n=30_000)
        rr.log("world/target_mean",
               rr.Points3D(target_sub, colors=shaded(target_sub, blue, lo, hi), radii=0.012))
        rr.log("world/adapter_pred",
               rr.Points3D(pred_sub, colors=shaded(pred_sub, red, lo, hi), radii=0.012))
        rr.log("world/input_pts",
               rr.Points3D(input_sub, colors=shaded(input_sub, green, lo, hi), radii=0.008))
        rr.log("world/cameras",
               rr.Points3D(cameras, colors=shaded(cameras, purple, lo, hi), radii=0.04))
        rr.log("legend", rr.TextDocument(
            f"# Online variance Hungarian validation start={start} ({i + 1}/{len(loaded)})\n\n"
            "- **Blue** target distribution mean\n"
            "- **Red** adapter prediction\n"
            "- **Green** input point sample\n"
            "- **Purple** cameras\n\n"
            f"Checkpoint step: `{ckpt_data.get('step')}`\n\n"
            f"Weighted val loss: `{ckpt_data.get('val_loss')}`\n\n"
            f"Plain val Hungarian MSE: `{metadata.get('val_plain_hungarian_mse')}`\n\n"
            f"Root: `{root}`\n\n"
            f"Frames: `{meta['frame_ids'][0]}-{meta['frame_ids'][-1]}`",
            media_type=rr.MediaType.MARKDOWN,
        ))

    print(f"[rerun] saved -> {args.rrd_out}")


if __name__ == "__main__":
    main()
