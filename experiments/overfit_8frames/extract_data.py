#!/usr/bin/env python3
"""
Step 1 — Extract and save 8-frame data from Replica room0.

NOVA3R AE input (z*):
  For each of the 8 frames, find the subset of GT mesh points visible from
  that frame using the GT depth map + crop_visible_world_points, then
  project them all to the first-camera coordinate frame and concatenate.
  This concatenated cloud is encoded by the frozen NOVA3R scene_ae.

DA3 source tokens:
  All 8 frames are passed through the frozen DA3 backbone.
  Per-frame tokens are subsampled and concatenated → (1, T*N, D).

Saves to <data_dir>/:
  da3_tokens.pt   (1, T*N_tokens, D_da3)
  z_star.pt       (1, 768, 128)
  meta.pt         frame_ids, poses, intrinsics, …

Usage:
  python extract_data.py [--config config.yaml] [--stride 20] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from safetensors.torch import load_file
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.inference import normalize_input                      # noqa: E402
from depth_anything_3.cfg import create_object                   # noqa: E402
from vc3r.replica import (                      # noqa: E402
    crop_visible_world_points,
)


# ── DA3 loading ────────────────────────────────────────────────────────────────

def load_da3_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if not model_name.startswith("depth-anything/"):
        snapshot = Path(model_name)
    else:
        repo = model_name.replace("/", "--")
        # Try repo-local HF cache first, then user home cache
        for cache_root in [REPO_ROOT / ".hf_cache", Path.home() / ".cache" / "huggingface"]:
            model_dir = cache_root / "hub" / f"models--{repo}"
            if model_dir.exists():
                break
        revision = (model_dir / "refs" / "main").read_text().strip()
        snapshot  = model_dir / "snapshots" / revision

    with (snapshot / "config.json").open() as f:
        payload = json.load(f)

    model = create_object(OmegaConf.create(payload["config"]))
    state = load_file(str(snapshot / "model.safetensors"), device="cpu")
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    """images: (B, T, 3, H, W) in [0,1]"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
    std  = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
    return (images - mean[None, None, :, None, None]) / std[None, None, :, None, None]


def normalize_extrinsics(poses_c2w: torch.Tensor) -> torch.Tensor:
    """Normalize so first camera is at origin; scale by median translation."""
    transform  = torch.linalg.inv(poses_c2w[:, :1])   # (1, 4, 4)
    normalized = poses_c2w @ transform                  # (B, T, 4, 4)
    c2ws       = torch.linalg.inv(normalized)
    translations = c2ws[..., :3, 3]
    median_dist  = translations.norm(dim=-1).median().clamp(min=1e-1)
    normalized[..., :3, 3] = normalized[..., :3, 3] / median_dist
    return normalized


def select_tokens(tokens: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n <= 0 or tokens.shape[1] <= max_n:
        return tokens
    idx = torch.linspace(0, tokens.shape[1] - 1, max_n, device=tokens.device).long()
    return tokens[:, idx]


@torch.no_grad()
def extract_da3_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,        # (T, 3, H, W) float32 in [0,1]
    poses_c2w: torch.Tensor,     # (T, 4, 4)
    intrinsics: torch.Tensor,    # (T, 3, 3)
    source_layer_index: int,
    max_source_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns (1, T*max_source_tokens, D_da3)."""
    images_bt = images.unsqueeze(0).to(device)         # (1, T, 3, H, W)
    poses_bt  = poses_c2w.unsqueeze(0).to(device)      # (1, T, 4, 4)
    K_bt      = intrinsics.unsqueeze(0).to(device)     # (1, T, 3, 3)

    images_norm     = imagenet_normalize(images_bt)
    extrinsics_w2c  = normalize_extrinsics(torch.linalg.inv(poses_bt[0]).unsqueeze(0))

    amp_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)

    with torch.autocast(device_type=device.type, enabled=False):
        cam_token = model.cam_enc(extrinsics_w2c, K_bt, images_bt.shape[-2:])

    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                        enabled=(device.type == "cuda")):
        backbone_out, _ = model.backbone(images_norm, cam_token=cam_token)

    # backbone_out[layer_idx] is a tuple; [0] = patch features (B*T, H_p, W_p, D)
    raw  = backbone_out[source_layer_index][0].float()    # (T, H_p, W_p, D)
    flat = raw.reshape(raw.shape[0], -1, raw.shape[-1])   # (T, P, D)

    per_frame = select_tokens(flat, max_source_tokens)    # (T, N, D)
    combined  = per_frame.reshape(1, -1, per_frame.shape[-1])  # (1, T*N, D)
    return combined.cpu()


# ── visible-point extraction ───────────────────────────────────────────────────

def world_to_first_camera(pts_world: torch.Tensor, first_pose_c2w: torch.Tensor) -> torch.Tensor:
    w2c  = torch.linalg.inv(first_pose_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


def sample_points(pts: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    if pts.shape[0] >= count:
        idx = torch.randperm(pts.shape[0], generator=gen)[:count]
    else:
        pad = torch.randint(pts.shape[0], (count - pts.shape[0],), generator=gen)
        idx = torch.cat([torch.arange(pts.shape[0]), pad])
    return pts[idx], idx


@torch.no_grad()
def encode_nova3r_ae(
    model, cfg,
    pts_first_cam: torch.Tensor,   # (1, N, 3)
    device: torch.device,
) -> torch.Tensor:
    """Returns z_star (1, K, D)."""
    pts   = pts_first_cam.to(device).float()
    valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)
    norm_mode = cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
    encoder_data = model._encode(pointmaps=pts_norm, test=True)
    return encoder_data["tokens"].float().cpu()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 8-frame DA3 tokens + NOVA3R AE z* from Replica"
    )
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override frame_stride from config")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override data_dir (default: <cfg.data_dir>/stride_<N>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if data already exists")
    parser.add_argument("--depth-tolerance", type=float, default=0.05,
                        help="Depth tolerance for crop_visible_world_points")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.stride is not None:
        cfg.frame_stride = args.stride

    device   = torch.device(args.device)
    # auto-namespace by stride so multiple runs don't clobber each other
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = Path(cfg.data_dir) / f"stride_{int(cfg.frame_stride)}"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_da3    = data_dir / "da3_tokens.pt"
    out_zstar  = data_dir / "z_star.pt"
    out_meta   = data_dir / "meta.pt"

    if not args.force and all(p.exists() for p in [out_da3, out_zstar, out_meta]):
        print(f"[extract] Data already at {data_dir}. Use --force to re-extract.")
        return

    # ── scene paths ──────────────────────────────────────────────────────────
    replica_root = Path(cfg.replica_root)
    room_dir     = replica_root / cfg.room
    results_dir  = room_dir / "results"
    mesh_ply     = replica_root / f"{cfg.room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]

    depth_scale = float(cam["scale"])
    H_native    = int(cam["h"])
    W_native    = int(cam["w"])
    K_native    = np.array([
        [cam["fx"], 0.0,       cam["cx"]],
        [0.0,       cam["fy"], cam["cy"]],
        [0.0,       0.0,       1.0      ],
    ], dtype=np.float32)

    poses_all  = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    frame_ids   = [int(p.stem.replace("frame", "")) for p in frame_files]
    assert len(frame_ids) == len(poses_all), "frame/pose count mismatch"

    stride   = int(cfg.frame_stride)
    n_frames = int(cfg.num_frames)
    assert len(frame_ids) >= (n_frames - 1) * stride + 1, \
        f"Not enough frames for {n_frames} frames at stride {stride}"

    sampled_ids   = [frame_ids[i * stride] for i in range(n_frames)]
    sampled_poses = np.stack([poses_all[fid] for fid in sampled_ids])   # (T, 4, 4)

    print(f"[extract] Room   : {cfg.room}")
    print(f"[extract] Frames : {sampled_ids}  (stride={stride})")
    print(f"[extract] Device : {device}")

    # ── load and resize RGB + depth ──────────────────────────────────────────
    H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
    scale_x = W_proc / W_native
    scale_y = H_proc / H_native
    K_proc  = K_native.copy()
    K_proc[0] *= scale_x
    K_proc[1] *= scale_y

    def load_rgb(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img = img.resize((W_proc, H_proc), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

    def load_depth_native(fid: int) -> torch.Tensor:
        """Load depth at native Replica resolution (for point projection)."""
        depth_img = Image.open(results_dir / f"depth{fid:06d}.png")
        return torch.from_numpy(
            np.asarray(depth_img, dtype=np.float32) / depth_scale
        )   # (H_native, W_native)

    images     = torch.stack([load_rgb(fid) for fid in sampled_ids])     # (T, 3, H, W)
    poses_t    = torch.from_numpy(sampled_poses).float()                   # (T, 4, 4)
    intrinsics = torch.from_numpy(
        np.tile(K_proc[None], (n_frames, 1, 1))
    ).float()                                                               # (T, 3, 3)
    K_native_t = torch.from_numpy(K_native)                                # (3, 3)

    # ── 1. Sample GT mesh points ─────────────────────────────────────────────
    print(f"[extract] Loading mesh: {mesh_ply}")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))           # (2M, 3)
    print(f"[extract] Mesh points sampled: {mesh_pts.shape[0]:,}")

    # ── 2. Per-frame visible points → project to first-camera frame ──────────
    first_pose_c2w  = poses_t[0]
    points_per_frame_list: list[torch.Tensor] = []

    for i, fid in enumerate(sampled_ids):
        depth = load_depth_native(fid)                                    # (H_nat, W_nat)
        c2w   = poses_t[i]                                                # (4, 4)
        visible = crop_visible_world_points(
            points_world   = mesh_pts,
            camera_to_world= c2w,
            intrinsics     = K_native_t,
            depth          = depth,
            depth_tolerance= args.depth_tolerance,
        )
        pts_world = visible["points_world"]                               # (V, 3)
        pts_cam   = world_to_first_camera(pts_world, first_pose_c2w)     # (V, 3)
        points_per_frame_list.append(pts_cam)
        print(f"  frame {fid:06d}: {pts_world.shape[0]:>6} visible pts"
              f" → {pts_cam.shape[0]:>6} in first-cam frame")

    all_visible = torch.cat(points_per_frame_list, dim=0)                 # (V_total, 3)
    print(f"[extract] Total visible pts: {all_visible.shape[0]:,}")

    # Subsample to token_input_points
    token_pts, sampled_idx = sample_points(
        all_visible, int(cfg.mesh_sample_points), seed=0
    )                                                                      # (N, 3)
    token_pts_batch = token_pts.unsqueeze(0)                              # (1, N, 3)
    print(f"[extract] Subsampled to: {token_pts.shape[0]} points for NOVA3R AE")

    # ── 3. Encode with NOVA3R AE ──────────────────────────────────────────────
    print("[extract] Loading NOVA3R AE ...")
    nova3r_model, nova3r_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova3r_model.eval()
    for p in nova3r_model.parameters():
        p.requires_grad_(False)

    print("[extract] Encoding visible points → z* ...")
    z_star = encode_nova3r_ae(nova3r_model, nova3r_cfg, token_pts_batch, device)
    print(f"[extract] z_star shape: {tuple(z_star.shape)}")

    del nova3r_model
    torch.cuda.empty_cache()

    # ── 4. Extract DA3 tokens ─────────────────────────────────────────────────
    print("[extract] Loading DA3 ...")
    da3_model = load_da3_model(cfg.da3_model, device)

    print("[extract] Extracting DA3 backbone tokens ...")
    da3_tokens = extract_da3_tokens(
        model             = da3_model,
        images            = images,
        poses_c2w         = poses_t,
        intrinsics        = intrinsics,
        source_layer_index= int(cfg.da3_source_layer_index),
        max_source_tokens = int(cfg.da3_max_source_tokens),
        device            = device,
    )   # (1, T*N, D)
    print(f"[extract] DA3 tokens shape: {tuple(da3_tokens.shape)}")

    del da3_model
    torch.cuda.empty_cache()

    # ── 5. Save ───────────────────────────────────────────────────────────────
    torch.save(da3_tokens, out_da3)
    torch.save(z_star, out_zstar)
    torch.save({
        "frame_ids":     sampled_ids,
        "poses_c2w":     poses_t,
        "intrinsics_proc": intrinsics,
        "K_native":      K_native_t,
        "stride":        stride,
        "num_frames":    n_frames,
        "room":          cfg.room,
        "image_hw_proc": (H_proc, W_proc),
        "image_hw_native": (H_native, W_native),
        "depth_tolerance": args.depth_tolerance,
        "norm_mode":     str(
            nova3r_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
        ),
        "token_input_points": token_pts.shape[0],
        "sampled_indices": sampled_idx,
    }, out_meta)

    print(f"\n[extract] Saved:")
    print(f"  {out_da3}    → {tuple(da3_tokens.shape)}")
    print(f"  {out_zstar}  → {tuple(z_star.shape)}")
    print(f"  {out_meta}")


if __name__ == "__main__":
    main()
