#!/usr/bin/env python3
"""Stage one ScanNet++ iPhone sequence for stitch_office4.py.

Reads RGB video and metadata directly from the shared dataset mirror. Only the
uniformly sampled RGB frames, poses, and one median intrinsic matrix are written
to the requested work directory. Sensor depth is unnecessary because the stitch
evaluation uses --complete-target and the laser-scan mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mirror", type=Path,
                   default=Path("/storage/group/dataset_mirrors/01_incoming/scannetpp"))
    p.add_argument("--scene", required=True, help="ScanNet++ scene ID")
    p.add_argument("--out", type=Path, required=True,
                   help="Temporary Replica-style dataset root")
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--width", type=int, default=960,
                   help="Staged RGB width; aspect ratio is preserved")
    return p.parse_args()


def main():
    args = parse_args()
    scene = args.mirror / "data" / args.scene
    video = scene / "iphone" / "rgb.mkv"
    metadata_path = scene / "iphone" / "pose_intrinsic_imu.json"
    mesh = scene / "scans" / "mesh_aligned_0.05.ply"
    for path in (video, metadata_path, mesh):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text())
    keys = sorted(metadata)
    if not keys:
        raise RuntimeError(f"No pose metadata in {metadata_path}")
    n = min(args.n_frames, len(keys))
    source_ids = np.linspace(0, len(keys) - 1, n).round().astype(int)
    source_ids = np.unique(source_ids)
    wanted = {int(src): out for out, src in enumerate(source_ids)}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_frames != len(keys):
        raise RuntimeError(f"RGB/metadata frame mismatch: {video_frames} != {len(keys)}")
    out_w = min(args.width, src_w)
    out_h = round(src_h * out_w / src_w)
    scale_x, scale_y = out_w / src_w, out_h / src_h

    results = args.out / args.scene / "results"
    results.mkdir(parents=True, exist_ok=True)
    poses = []
    intrinsics = []
    written = 0
    source_idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        out_idx = wanted.get(source_idx)
        if out_idx is not None:
            if (out_w, out_h) != (src_w, src_h):
                image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
            dst = results / f"frame{out_idx:06d}.jpg"
            if not cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write {dst}")
            entry = metadata[keys[source_idx]]
            poses.append(np.asarray(entry["aligned_pose"], dtype=np.float64))
            K = np.asarray(entry["intrinsic"], dtype=np.float64).copy()
            K[0] *= scale_x
            K[1] *= scale_y
            intrinsics.append(K)
            written += 1
        source_idx += 1
    cap.release()
    if written != len(source_ids):
        raise RuntimeError(f"Decoded {written}/{len(source_ids)} requested frames")

    np.savetxt(args.out / args.scene / "traj.txt",
               np.stack(poses).reshape(-1, 16), fmt="%.10f")
    K = np.median(np.stack(intrinsics), axis=0)
    camera = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
              "cx": float(K[0, 2]), "cy": float(K[1, 2]),
              "w": out_w, "h": out_h, "scale": 1000.0}
    (args.out / "cam_params.json").write_text(json.dumps({"camera": camera}, indent=2))
    print(f"[stage] {args.scene}: {written}/{video_frames} frames at {out_w}x{out_h}")
    print(f"[stage] data root: {args.out}")
    print(f"[stage] GT mesh remains in mirror: {mesh}")


if __name__ == "__main__":
    main()
