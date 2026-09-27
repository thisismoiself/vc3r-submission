#!/usr/bin/env python3
"""Materialize the NeuralRGBD dataset into the Replica directory layout.

Rationale: the z_star caching path (cache_online_hungarian_zstar_windows.py) is proven
and its target distribution is what the adapter learns. Rather than teach it a second
dataset, we convert NeuralRGBD to look exactly like Replica, so the identical recipe runs
untouched and z_star targets stay in one distribution.

Verified empirically against the GT mesh (thin_geometry, frames 50/100/200/300):
  * poses.txt is ground truth in OpenGL convention -> needs @ diag(1,-1,-1,1) to reach
    the OpenCV (+Z forward) convention that crop_visible_world_points assumes.
    Without the flip, unprojected depth lands ~70cm off the mesh instead of ~0.5cm.
  * depth/ is uint16 millimetres (scale 1000), storing z-depth (not ray distance).
  * All 9 scenes share focal 554.2562584220408 at 640x480, so one cam_params.json serves
    the whole root and no cache-script change is required.

We take poses.txt (ground truth), NOT trainval_poses.txt (the pose set the NeuralRGBD
optimization consumes) -- the mesh/depth visibility check needs true poses.
We take depth/ (GT depth for these synthetic scenes), NOT depth_with_noise/.
We take gt_mesh.ply (full), NOT gt_mesh_culled.ply -- the visibility crop does its own
culling, and the full mesh matches Replica's <room>_mesh.ply.

Usage:
  python scripts/convert_neural_rgbd.py \
      --src /usr/prakt/s0016/NeuralRGBD_dl/raw \
      --meshes /usr/prakt/s0016/NeuralRGBD_dl/raw_meshes \
      --out /usr/prakt/s0016/NeuralRGBD
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

# NeuralRGBD ships OpenGL camera-to-world (-Z forward, +Y up); Replica/our pipeline
# expects OpenCV (+Z forward, +Y down).
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

SCENES = [
    "breakfast_room", "complete_kitchen", "green_room", "grey_white_room",
    "kitchen", "morning_apartment", "staircase", "thin_geometry", "whiteroom",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="extracted neural_rgbd_data root")
    p.add_argument("--meshes", type=Path, required=True, help="extracted meshes.zip root")
    p.add_argument("--out", type=Path, required=True, help="output Replica-layout root")
    p.add_argument("--scenes", nargs="*", default=SCENES)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--depth-scale", type=float, default=1000.0)
    return p.parse_args()


def numeric_index(path: Path, prefix: str) -> int:
    m = re.fullmatch(rf"{prefix}(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Unexpected filename: {path}")
    return int(m.group(1))


def convert_scene(scene: str, args: argparse.Namespace) -> dict:
    src = args.src / scene
    out_scene = args.out / scene
    results = out_scene / "results"
    results.mkdir(parents=True, exist_ok=True)

    imgs = sorted(src.glob("images/img*.png"), key=lambda p: numeric_index(p, "img"))
    deps = sorted(src.glob("depth/depth*.png"), key=lambda p: numeric_index(p, "depth"))
    img_ids = [numeric_index(p, "img") for p in imgs]
    dep_ids = [numeric_index(p, "depth") for p in deps]
    if img_ids != dep_ids:
        raise ValueError(f"{scene}: image/depth frame ids differ")
    if img_ids != list(range(len(img_ids))):
        raise ValueError(f"{scene}: frame ids are not contiguous from 0")

    poses = np.loadtxt(src / "poses.txt", dtype=np.float64).reshape(-1, 4, 4)
    if len(poses) != len(imgs):
        raise ValueError(f"{scene}: {len(poses)} poses vs {len(imgs)} frames")

    # GL -> CV, then flatten to Replica's 16-values-per-row traj.txt.
    poses_cv = poses @ GL2CV
    np.savetxt(out_scene / "traj.txt", poses_cv.reshape(-1, 16), fmt="%.10f")

    for fid, (ip, dp) in enumerate(zip(imgs, deps)):
        Image.open(ip).convert("RGB").save(
            results / f"frame{fid:06d}.jpg", quality=args.jpeg_quality
        )
        dst = results / f"depth{fid:06d}.png"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(dp.resolve())

    mesh_src = (args.meshes / scene / "gt_mesh.ply").resolve()
    if not mesh_src.exists():
        raise FileNotFoundError(f"{scene}: missing {mesh_src}")
    mesh_dst = args.out / f"{scene}_mesh.ply"
    if mesh_dst.is_symlink() or mesh_dst.exists():
        mesh_dst.unlink()
    mesh_dst.symlink_to(mesh_src)

    return {"scene": scene, "frames": len(imgs)}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    focals = {
        s: float((args.src / s / "focal.txt").read_text().strip()) for s in args.scenes
    }
    if len(set(focals.values())) != 1:
        raise ValueError(f"Scenes disagree on focal, need per-scene cam_params: {focals}")
    focal = next(iter(focals.values()))

    with Image.open(next((args.src / args.scenes[0]).glob("images/img*.png"))) as im:
        w, h = im.size

    cam = {
        "camera": {
            "fx": focal, "fy": focal, "cx": w / 2.0, "cy": h / 2.0,
            "w": w, "h": h, "scale": args.depth_scale,
        }
    }
    (args.out / "cam_params.json").write_text(json.dumps(cam, indent=2))
    print(f"[convert] cam_params.json  focal={focal:.4f}  {w}x{h}  scale={args.depth_scale}")

    total = 0
    for scene in args.scenes:
        info = convert_scene(scene, args)
        total += info["frames"]
        print(f"[convert] {scene:<18} {info['frames']:>5} frames")
    print(f"[convert] done: {len(args.scenes)} scenes, {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
