#!/usr/bin/env python3
"""Export point-cloud error visualization to a Rerun .rrd file."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "input": np.array([150, 150, 150], dtype=np.uint8),
    "target": np.array([60, 130, 220], dtype=np.uint8),
    "prediction": np.array([230, 80, 65], dtype=np.uint8),
}


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing PLY end_header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        if header_lines[0] != "ply" or "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"{path}: expected binary little-endian PLY")

        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError(f"{path}: missing vertex count")

        raw = f.read(vertex_count * 15)
        if len(raw) != vertex_count * 15:
            raise ValueError(f"{path}: unexpected payload size")

    xyz = np.empty((vertex_count, 3), dtype=np.float32)
    rgb = np.empty((vertex_count, 3), dtype=np.uint8)
    for i in range(vertex_count):
        offset = i * 15
        x, y, z, r, g, b = struct.unpack_from("<fffBBB", raw, offset)
        xyz[i] = (x, y, z)
        rgb[i] = (r, g, b)
    return xyz, rgb


def make_legend_image() -> np.ndarray:
    rows = [
        ("heldout/input", "Input point cloud", COLORS["input"]),
        ("heldout/target_zstar_decode", "Target z_star decode", COLORS["target"]),
        ("heldout/prediction", "Adapter prediction", COLORS["prediction"]),
    ]
    width, height = 760, 190
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font_title = font = font_small = ImageFont.load_default()

    draw.text((18, 14), "Point cloud legend", fill=(20, 20, 20), font=font_title)
    y = 58
    for entity, label, color in rows:
        rgb = tuple(int(c) for c in color)
        draw.rounded_rectangle((22, y, 58, y + 28), radius=4, fill=rgb, outline=(40, 40, 40))
        draw.text((72, y - 1), label, fill=(20, 20, 20), font=font)
        draw.text((320, y + 2), entity, fill=(80, 80, 80), font=font_small)
        y += 42
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.01)
    args = parser.parse_args()

    input_xyz, input_rgb = read_ply_xyz_rgb(args.input_dir / "input_cloud.ply")
    gt_xyz, gt_rgb = read_ply_xyz_rgb(args.input_dir / "gt_decoded.ply")
    pred_xyz, pred_rgb = read_ply_xyz_rgb(args.input_dir / "pred_decoded_error_colored.ply")
    metrics = (args.input_dir / "metrics.txt").read_text(encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("da3_nova3r_pointcloud_error", spawn=False)
    rr.save(str(args.out))

    rr.log("metrics", rr.TextDocument(metrics, media_type=rr.MediaType.MARKDOWN))
    rr.log("heldout", rr.ViewCoordinates.RDF, static=True)
    rr.log("overlay", rr.ViewCoordinates.RDF, static=True)

    input_color = np.tile(COLORS["input"], (len(input_xyz), 1))
    target_color = np.tile(COLORS["target"], (len(gt_xyz), 1))
    prediction_color = np.tile(COLORS["prediction"], (len(pred_xyz), 1))

    rr.log("legend", rr.Image(make_legend_image()))
    rr.log(
        "legend_text",
        rr.TextDocument(
            "\n".join([
                "# Point cloud legend",
                "",
                "- grey: `heldout/input`",
                "- blue: `heldout/target_zstar_decode`",
                "- red: `heldout/prediction`",
            ]),
            media_type=rr.MediaType.MARKDOWN,
        ),
    )
    rr.log(
        "heldout/input",
        rr.Points3D(input_xyz, colors=input_color, radii=args.radius * 0.45),
    )
    rr.log(
        "heldout/target_zstar_decode",
        rr.Points3D(gt_xyz, colors=target_color, radii=args.radius),
    )
    rr.log(
        "heldout/prediction",
        rr.Points3D(pred_xyz, colors=prediction_color, radii=args.radius),
    )

    rr.log(
        "overlay/target_zstar_decode",
        rr.Points3D(gt_xyz, colors=target_color, radii=args.radius),
    )
    rr.log(
        "overlay/prediction",
        rr.Points3D(pred_xyz, colors=prediction_color, radii=args.radius),
    )

    print(f"Wrote {args.out}")
    print("Open with:")
    print(f"  rerun {args.out}")


if __name__ == "__main__":
    main()
