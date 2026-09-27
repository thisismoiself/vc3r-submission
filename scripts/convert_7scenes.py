#!/usr/bin/env python3
"""Convert a 7-Scenes sequence to the Replica-style layout the NOVA3R adapter stitcher expects:
  <out>/cam_params.json                     (7-Scenes: fx=fy=585, cx=320, cy=240, 640x480, mm depth)
  <out>/<room>/results/frame{i:06d}.jpg     (RGB)
  <out>/<room>/results/depth{i:06d}.png     (uint16 mm depth)
  <out>/<room>/traj.txt                     (N x 16, camera-to-world, OpenCV)
  <out>/<room>_mesh.ply                     (Open3D TSDF fusion of depth+poses = the fused-scan "GT")
7-Scenes has no complete mesh; the TSDF fusion of its own RGB-D scan is the ground-truth-equivalent
(incomplete, real). Frames are subsampled to keep the trajectory manageable."""
import os, sys, json, argparse, shutil
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="7-Scenes sequence dir (has frame-*.color.png)")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--room", required=True, help="scene name, e.g. heads")
    ap.add_argument("--every", type=int, default=5, help="keep every Nth frame")
    ap.add_argument("--fx", type=float, default=585.0)
    ap.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size (m)")
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out); room = args.room
    res = out / room / "results"; res.mkdir(parents=True, exist_ok=True)
    fids = sorted(int(p.stem.split("-")[1].split(".")[0]) for p in src.glob("frame-*.color.png"))
    fids = fids[::args.every]
    print(f"[7scenes] {room}: {len(fids)} frames (every {args.every})", flush=True)

    K = dict(fx=args.fx, fy=args.fx, cx=320.0, cy=240.0, w=640, h=480, scale=1000.0)
    json.dump({"camera": K}, open(out / "cam_params.json", "w"), indent=2)

    import open3d as o3d
    intr = o3d.camera.PinholeCameraIntrinsic(640, 480, args.fx, args.fx, 320.0, 240.0)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=5 * args.voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    poses = []
    for i, f in enumerate(fids):
        color = Image.open(src / f"frame-{f:06d}.color.png").convert("RGB")
        depth = np.array(Image.open(src / f"frame-{f:06d}.depth.png"))          # uint16 mm
        depth = depth.copy(); depth[depth == 65535] = 0                          # 65535 = invalid
        pose = np.loadtxt(src / f"frame-{f:06d}.pose.txt").astype(np.float64)    # c2w
        if not np.isfinite(pose).all():
            continue
        color.save(res / f"frame{i:06d}.jpg", quality=95)
        Image.fromarray(depth.astype(np.uint16)).save(res / f"depth{i:06d}.png")
        poses.append(pose.reshape(-1))
        # TSDF integrate (needs world->camera extrinsic)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(color)),
            o3d.geometry.Image(depth.astype(np.uint16)),
            depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(pose))
    np.savetxt(out / f"{room}/traj.txt", np.stack(poses), fmt="%.8f")

    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / f"{room}_mesh.ply"), mesh)
    print(f"[7scenes] wrote {len(poses)} frames + mesh ({len(mesh.vertices):,} verts) -> {out}/{room}_mesh.ply", flush=True)


if __name__ == "__main__":
    main()
