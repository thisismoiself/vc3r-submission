#!/usr/bin/env python3
"""
Export decoded NOVA3R point clouds as coloured .ply files for CloudCompare.

For each sample the script produces three files:
  sample_NNN_input.ply      – frustum-sampled mesh points coloured from RGB images
  sample_NNN_gt_decoded.ply – NOVA3R AE round-trip (z_star → ODE → pts)
  sample_NNN_pred_decoded.ply – adapter prediction (DA3 → adapter → ODE → pts)

The decoded outputs inherit colours from the nearest input point so that all
three clouds share a consistent colour palette in CloudCompare.

Usage:
  python export_ply.py --n-samples 50 --out-dir ply_export/N50
  python export_ply.py --n-samples 50 --out-dir ply_export/N50 --samples 0 1 2
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT   = Path(__file__).resolve().parents[2]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.inference import normalize_input, amp_dtype_mapping  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from depth_anything_3.cfg import create_object                             # noqa: E402
from safetensors.torch import load_file                                     # noqa: E402


# ── PLY writer ────────────────────────────────────────────────────────────────

def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    """Write binary little-endian PLY with xyz and optional uint8 rgb."""
    path.parent.mkdir(parents=True, exist_ok=True)
    N = xyz.shape[0]
    has_color = rgb is not None
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if has_color:
        header += (
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
        )
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if has_color:
            rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
            for i in range(N):
                f.write(struct.pack("<fff", *xyz[i].astype(np.float32)))
                f.write(struct.pack("BBB", *rgb8[i]))
        else:
            f.write(xyz.astype(np.float32).tobytes())


# ── Colour a point cloud from RGB images with depth testing ──────────────────

def colour_points_from_images(pts_first_cam: np.ndarray,
                               poses_c2w: torch.Tensor,
                               intrinsics_nat: np.ndarray,
                               image_hw: tuple,
                               results_dir: Path,
                               frame_ids: list,
                               depth_scale: float,
                               depth_tolerance: float = 0.05) -> np.ndarray:
    """
    Project pts_first_cam (in first-camera metric frame) onto each camera in
    the window. Assigns the colour from the camera where the point projects
    most centrally AND passes the depth test (|z - depth_map[v,u]| < tol).
    Occluded or out-of-frustum points get grey (128, 128, 128).
    """
    H, W = image_hw
    fx, fy = intrinsics_nat[0, 0], intrinsics_nat[1, 1]
    cx, cy = intrinsics_nat[0, 2], intrinsics_nat[1, 2]
    N = pts_first_cam.shape[0]

    first_c2w = poses_c2w[0].numpy().astype(np.float64)
    ones = np.ones((N, 1), dtype=np.float64)
    pts_world = (np.concatenate([pts_first_cam.astype(np.float64), ones], axis=1)
                 @ first_c2w.T)[:, :3]
    pts_world_h = np.concatenate([pts_world, ones], axis=1)

    cx_img, cy_img = (W - 1) / 2.0, (H - 1) / 2.0
    best_score = np.full(N, -1e9, dtype=np.float64)
    best_uv    = np.zeros((N, 2), dtype=np.float32)
    best_fid   = np.full(N, -1, dtype=np.int32)

    for cam_idx, fid in enumerate(frame_ids):
        w2c = np.linalg.inv(poses_c2w[cam_idx].numpy().astype(np.float64))
        pts_cam = (pts_world_h @ w2c.T)[:, :3]
        z = pts_cam[:, 2]
        u = np.where(z > 0, pts_cam[:, 0] / z * fx + cx, -1.0)
        v = np.where(z > 0, pts_cam[:, 1] / z * fy + cy, -1.0)
        in_image = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_image):
            continue

        depth_img = (np.asarray(Image.open(results_dir / f"depth{fid:06d}.png"),
                                dtype=np.float32) / depth_scale)
        u_i = np.clip(u.astype(int), 0, W - 1)
        v_i = np.clip(v.astype(int), 0, H - 1)
        depth_ok = in_image & (np.abs(z - depth_img[v_i, u_i]) < depth_tolerance)

        score = np.where(depth_ok, -((u - cx_img) ** 2 + (v - cy_img) ** 2), -1e9)
        update = score > best_score
        best_score[update] = score[update]
        best_uv[update, 0] = u[update]
        best_uv[update, 1] = v[update]
        best_fid[update]   = fid

    colours = np.full((N, 3), 128, dtype=np.uint8)   # grey for occluded / no-hit
    for fid in [f for f in np.unique(best_fid) if f >= 0]:
        img  = np.asarray(Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB"),
                          dtype=np.uint8)
        mask = best_fid == fid
        us = np.clip(best_uv[mask, 0].astype(int), 0, W - 1)
        vs = np.clip(best_uv[mask, 1].astype(int), 0, H - 1)
        colours[mask] = img[vs, us]

    return colours


def transfer_colours_nn(source_pts: np.ndarray, source_rgb: np.ndarray,
                         target_pts: np.ndarray) -> np.ndarray:
    """Assign each target point the colour of its nearest source point."""
    from scipy.spatial import KDTree
    tree = KDTree(source_pts)
    _, idx = tree.query(target_pts, workers=-1)
    return source_rgb[idx]


# ── NOVA3R decode ──────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens, pts_norm, device,
                  num_queries=8192, seed=42):
    torch.manual_seed(seed)
    B = tokens.shape[0]
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(B, num_queries, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    with torch.cuda.amp.autocast(enabled=device.type != "cpu", dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    pts3d = sol[-1] if isinstance(sol, list) else sol
    return pts3d.cpu()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",   type=int, required=True)
    parser.add_argument("--stride",      type=int, default=20)
    parser.add_argument("--out-dir",     type=Path, default=None)
    parser.add_argument("--samples",     type=int, nargs="*", default=None,
                        help="Which sample indices to export (default: all)")
    parser.add_argument("--num-queries", type=int, default=8192)
    parser.add_argument("--ckpt",        type=Path, default=None,
                        help="Override adapter checkpoint path")
    parser.add_argument("--config",      type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)
    N      = args.n_samples
    stride = args.stride

    EXP_DIR   = REPO_ROOT / "experiments" / "overfit_8frames"
    data_root = EXP_DIR / "data" / f"multi_N{N}_s{stride}"
    ckpt_path = args.ckpt or EXP_DIR / "checkpoints" / f"multi_N{N}_s{stride}" / "adapter_final.pt"
    out_dir   = args.out_dir or EXP_DIR / "ply_export" / f"N{N}_s{stride}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = args.samples if args.samples is not None else list(range(N))

    # ── scene metadata ────────────────────────────────────────────────────────
    replica_root = Path(str(cfg.replica_root))
    room         = str(cfg.room)
    room_dir    = replica_root / room
    results_dir = room_dir / "results"

    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat   = int(cam_p["h"]), int(cam_p["w"])
    depth_scale    = float(cam_p["scale"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                       [0., cam_p["fy"], cam_p["cy"]],
                       [0., 0., 1.]], dtype=np.float32)

    # ── load models ──────────────────────────────────────────────────────────
    print("[export_ply] Loading NOVA3R …")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters(): p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    print(f"[export_ply] Loading adapter from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()

    # ── per-sample export ─────────────────────────────────────────────────────
    for si in sample_indices:
        d = data_root / f"sample_{si:03d}"
        print(f"\n[export_ply] Sample {si:03d} …")

        da3_tokens = torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True)
        z_star     = torch.load(d / "z_star.pt",     map_location="cpu", weights_only=True)
        pts_norm   = torch.load(d / "pts_norm.pt",   map_location="cpu", weights_only=True)
        meta       = torch.load(d / "meta.pt",       map_location="cpu", weights_only=False)

        frame_ids = meta["frame_ids"]
        poses_c2w = meta["poses_c2w"]   # (T, 4, 4)

        # pts_norm[0] is the exact input cloud in first-camera metric frame
        # (norm_mode="none" → no scaling applied). All decoded outputs are in
        # the same coordinate space, so we project pts_norm[0] directly for colours.
        input_pts = pts_norm[0].numpy()   # (8192, 3)

        # -- colour input cloud via depth-tested image projection ------------
        print(f"  colouring {len(input_pts):,} input points …")
        input_rgb = colour_points_from_images(
            input_pts, poses_c2w, K_nat, (H_nat, W_nat),
            results_dir, frame_ids, depth_scale)

        # -- decode GT and predicted -----------------------------------------
        print("  decoding GT …")
        gt_dec  = decode_tokens(nova_model, nova_cfg, z_star,  pts_norm, device,
                                 args.num_queries, seed=42 + si)[0].numpy()
        print("  decoding predicted …")
        with torch.no_grad():
            pred_tokens = adapter(da3_tokens.to(device)).cpu()
        pred_dec = decode_tokens(nova_model, nova_cfg, pred_tokens, pts_norm, device,
                                  args.num_queries, seed=42 + si)[0].numpy()

        # Propagate colours to decoded clouds via nearest neighbour in the same space.
        gt_rgb   = transfer_colours_nn(input_pts, input_rgb, gt_dec)
        pred_rgb = transfer_colours_nn(input_pts, input_rgb, pred_dec)

        # -- write PLY files -------------------------------------------------
        stem = f"sample_{si:03d}"
        write_ply(out_dir / f"{stem}_input.ply",        input_pts, input_rgb)
        write_ply(out_dir / f"{stem}_gt_decoded.ply",   gt_dec,    gt_rgb)
        write_ply(out_dir / f"{stem}_pred_decoded.ply", pred_dec,  pred_rgb)
        print(f"  → {out_dir}/{stem}_*.ply")

    print(f"\n[export_ply] Done. {len(sample_indices)} sample(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
