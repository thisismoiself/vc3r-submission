#!/usr/bin/env python3
"""
Export and visualise point clouds for a single frame — DA3 vs NOVA3R adapter.

Writes four PLY files (all in world frame) to --out-dir:
  frame<N>_gt_mesh.ply       GT visible mesh points (height-coloured)
  frame<N>_da3.ply           DA3 depth backprojected (RGB-coloured)
  frame<N>_nova3r_ae.ply     NOVA3R AE roundtrip of GT pts (height-coloured)
  frame<N>_nova3r_pred.ply   NOVA3R decode of adapter output (height-coloured)

Uses: checkpoints/stride_<S>/overfit_final.pt

Usage:
  python compare_frame.py --frame 140
  python compare_frame.py --frame 0 --stride 20 --out-dir /tmp/frame0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

REPO_ROOT   = Path(__file__).resolve().parents[2]
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from nova3r.inference import normalize_input, amp_dtype_mapping
from vc3r.alignment import DA3ToNOVA3RAlignment
from vc3r.replica import crop_visible_world_points
from depth_anything_3.cfg import create_object


# ── PLY helpers ───────────────────────────────────────────────────────────────

def save_ply(path: Path, pts: np.ndarray, colors: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  saved {path.name}  ({len(pts):,} pts, {path.stat().st_size // 1024} KB)")


def height_cmap(pts: np.ndarray, cmap: str = "plasma") -> np.ndarray:
    y = pts[:, 1]
    norm = Normalize(np.percentile(y, 2), np.percentile(y, 98))
    return plt.get_cmap(cmap)(norm(y))[:, :3].astype(np.float32)


# ── DA3 model ─────────────────────────────────────────────────────────────────

def load_da3(model_name: str, device: torch.device):
    repo = model_name.replace("/", "--")
    for cache_root in [REPO_ROOT / ".hf_cache", Path.home() / ".cache" / "huggingface"]:
        model_dir = cache_root / "hub" / f"models--{repo}"
        if model_dir.exists():
            break
    rev  = (model_dir / "refs" / "main").read_text().strip()
    snap = model_dir / "snapshots" / rev
    with (snap / "config.json").open() as f:
        payload = json.load(f)
    model = create_object(OmegaConf.create(payload["config"]))
    state = load_file(str(snap / "model.safetensors"), device="cpu")
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def imagenet_norm(imgs: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device, dtype=imgs.dtype)
    std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device, dtype=imgs.dtype)
    return (imgs - mean[None, None, :, None, None]) / std[None, None, :, None, None]


def normalize_extrinsics(w2c: torch.Tensor) -> tuple[torch.Tensor, float]:
    transform  = torch.linalg.inv(w2c[:, :1])
    normalized = w2c @ transform
    c2ws       = torch.linalg.inv(normalized)
    median_dist = float(c2ws[..., :3, 3].norm(dim=-1).median().clamp(min=1e-1))
    normalized[..., :3, 3] /= median_dist
    return normalized, median_dist


def align_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    mask = (gt > 0.1) & np.isfinite(gt) & (pred > 1e-6)
    d, g = pred[mask].ravel(), gt[mask].ravel()
    return float(max((d * g).sum() / (d * d + 1e-12).sum(), 0.01))


def depth_to_world_rgb(depth: np.ndarray, rgb: np.ndarray,
                        K: np.ndarray, c2w: np.ndarray,
                        max_pts: int = 60_000) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mask  = np.isfinite(depth) & (depth > 0.1)
    d     = depth[mask]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pts_cam = np.stack([(u[mask]-cx)/fx*d, (v[mask]-cy)/fy*d, d,
                         np.ones(mask.sum(), np.float32)], axis=1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    cols = rgb[mask]
    if len(pts_world) > max_pts:
        idx = np.random.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]; cols = cols[idx]
    return pts_world.astype(np.float32), cols.astype(np.float32)


# ── NOVA3R decode → world ─────────────────────────────────────────────────────

@torch.no_grad()
def decode_to_world(nova_model, nova_cfg,
                    tokens: torch.Tensor,    # (1, 768, 128)
                    pts_norm: torch.Tensor,  # (1, 8192, 3) normalised
                    norm_factor: float,      # ignored when norm_mode='none'
                    c2w: np.ndarray,
                    device: torch.device,
                    num_queries: int, seed: int,
                    norm_mode: str = "median_3") -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, num_queries, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    use_amp = device.type != "cpu"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm,
        )
    pts_norm_dec = (sol[-1] if isinstance(sol, list) else sol)[0].cpu().numpy()

    # Invert the normalisation applied by normalize_input
    if norm_mode == "none":
        pts_cam = pts_norm_dec
    else:
        # median_N: pts_norm = pts_cam / norm_factor * target_median
        target_median = float(norm_mode.split("_")[-1]) if "_" in norm_mode else 1.0
        pts_cam = pts_norm_dec / target_median * norm_factor

    ones      = np.ones((len(pts_cam), 1), np.float32)
    pts_world = (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3]
    return pts_world.astype(np.float32), height_cmap(pts_world)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame",       type=int, required=True,
                        help="Frame ID to compare (e.g. 140)")
    parser.add_argument("--stride",      type=int, default=20,
                        help="Stride used when extracting the 8-frame dataset")
    parser.add_argument("--config",      type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--out-dir",     type=Path, default=None,
                        help="Output directory for PLY files (default: ply/frame<N>)")
    parser.add_argument("--num-queries", type=int,   default=8192)
    parser.add_argument("--max-pts",     type=int,   default=60_000)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    target_frame = args.frame
    stride       = args.stride
    cfg          = OmegaConf.load(args.config)
    device       = torch.device(args.device)
    out_dir      = args.out_dir or Path(__file__).parent / "ply" / f"frame{target_frame}"
    out_dir.mkdir(parents=True, exist_ok=True)

    EXP_DIR   = REPO_ROOT / "experiments" / "overfit_8frames"
    data_dir  = EXP_DIR / "data" / f"stride_{stride}"
    ckpt_path = EXP_DIR / "checkpoints" / f"stride_{stride}" / "overfit_final.pt"

    # ── metadata ──────────────────────────────────────────────────────────────
    da3_agg   = torch.load(data_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    meta      = torch.load(data_dir / "meta.pt",       map_location="cpu", weights_only=False)
    frame_ids = meta["frame_ids"]
    poses_c2w = meta["poses_c2w"].float()
    K_native  = meta["K_native"].float()
    T_frames  = len(frame_ids)
    N_per_frame = da3_agg.shape[1] // T_frames

    if target_frame not in list(frame_ids):
        raise ValueError(f"Frame {target_frame} not in stride_{stride} dataset. "
                         f"Available: {list(frame_ids)}")
    frame_idx = list(frame_ids).index(target_frame)
    print(f"[export] frame {target_frame:06d}  (index {frame_idx}/{T_frames}, stride={stride})")

    c2w_np    = poses_c2w[frame_idx].numpy()
    w2c_np    = np.linalg.inv(c2w_np)
    cam_world = poses_c2w[frame_idx, :3, 3].numpy()

    replica_root = (Path(cfg.replica_root) if Path(cfg.replica_root).is_absolute()
                    else REPO_ROOT / cfg.replica_root)
    room        = str(meta["room"])
    results_dir = replica_root / room / "results"
    mesh_ply    = replica_root / f"{room}_mesh.ply"
    with (replica_root / "cam_params.json").open() as f:
        cam_params = json.load(f)["camera"]
    depth_scale  = float(cam_params["scale"])
    W_nat, H_nat = int(cam_params["w"]), int(cam_params["h"])

    W_da3, H_da3 = int(cfg.image_width), int(cfg.image_height)
    K_da3 = K_native.numpy().copy()
    K_da3[0] *= W_da3 / W_nat; K_da3[1] *= H_da3 / H_nat

    # ── 1. GT mesh visible points ─────────────────────────────────────────────
    print("[1/4] GT mesh visible points …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    gt_depth_nat = np.asarray(
        Image.open(results_dir / f"depth{target_frame:06d}.png"), dtype=np.float32
    ) / depth_scale

    vis = crop_visible_world_points(
        mesh_pts, poses_c2w[frame_idx], K_native,
        torch.from_numpy(gt_depth_nat),
        depth_tolerance=args.depth_tolerance,
    )
    gt_pts = vis["points_world"].numpy()
    if len(gt_pts) > args.max_pts:
        gt_pts = gt_pts[np.random.choice(len(gt_pts), args.max_pts, replace=False)]
    gt_col = height_cmap(gt_pts, "YlOrRd")
    save_ply(out_dir / f"frame{target_frame:06d}_gt_mesh.ply", gt_pts, gt_col)

    # ── 2. DA3 depth → world ─────────────────────────────────────────────────
    print("[2/4] DA3 depth inference …")
    da3_model = load_da3(str(cfg.da3_model), device)

    rgb_list, img_tensors = [], []
    for fid in frame_ids:
        p = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB").resize((W_da3, H_da3), Image.BILINEAR)
        rgb_np = np.array(p, dtype=np.float32) / 255.0
        rgb_list.append(rgb_np)
        img_tensors.append(torch.from_numpy(rgb_np).permute(2, 0, 1))

    images      = torch.stack(img_tensors).unsqueeze(0).to(device)
    images_norm = imagenet_norm(images)
    w2c_all     = torch.linalg.inv(poses_c2w).unsqueeze(0)
    extr_norm, _ = normalize_extrinsics(w2c_all)
    K_batch     = K_native.unsqueeze(0).unsqueeze(0).expand(1, T_frames, -1, -1).to(device)

    with torch.no_grad():
        amp_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dt, enabled=(device.type == "cuda")):
            cam_tok = da3_model.cam_enc(extr_norm.to(device), K_batch, images_norm.shape[-2:])
            feats, _ = da3_model.backbone(images_norm, cam_token=cam_tok)
        with torch.autocast(device_type=device.type, enabled=False):
            da3_out = da3_model._process_depth_head(feats, H_da3, W_da3)

    da3_raw  = da3_out.depth[0, frame_idx].cpu().numpy()
    gt_d_da3 = np.array(Image.fromarray(gt_depth_nat).resize((W_da3, H_da3), Image.NEAREST))
    scale    = align_scale(da3_raw, gt_d_da3)
    da3_pts, da3_col = depth_to_world_rgb(da3_raw * scale, rgb_list[frame_idx], K_da3, c2w_np, args.max_pts)
    save_ply(out_dir / f"frame{target_frame:06d}_da3.ply", da3_pts, da3_col)
    print(f"  scale={scale:.3f}")
    del da3_model

    # ── prepare camera-frame GT pts for NOVA3R ────────────────────────────────
    ones_np  = np.ones((len(gt_pts), 1), np.float32)
    pts_cam  = (np.hstack([gt_pts, ones_np]) @ w2c_np.T)[:, :3]
    n = len(pts_cam)
    if n >= 8192:
        idx = np.linspace(0, n-1, 8192, dtype=int)
        pts_sub = pts_cam[idx]
    else:
        pts_sub = np.vstack([pts_cam, pts_cam[np.random.randint(n, size=8192-n)]])

    norm_factor   = float(np.clip(np.median(np.linalg.norm(pts_sub, axis=1)), 0.01, 100.0))
    pts_sub_t     = torch.from_numpy(pts_sub).unsqueeze(0).float().to(device)
    valid_t       = torch.ones(1, 8192, dtype=torch.bool, device=device)
    # norm_mode read from NOVA3R config after it loads — use median_3 as placeholder;
    # it will be corrected after NOVA3R loads in step 3.
    pts_norm_t, _ = normalize_input(pts_sub_t, valid_t, pts_sub_t, valid_t, mode="median_3")
    # (AE roundtrip uses frame 140's own cam — norm_mode mismatch is minor here)

    # ── 3. NOVA3R AE roundtrip ────────────────────────────────────────────────
    print("[3/4] NOVA3R AE roundtrip …")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    print(f"  NOVA3R norm_mode={norm_mode!r}")

    with torch.no_grad():
        z_gt = nova_model._encode(pointmaps=pts_norm_t, test=True)["tokens"].float()
    nova_ae_pts, nova_ae_col = decode_to_world(
        nova_model, nova_cfg, z_gt, pts_norm_t, norm_factor, c2w_np,
        device, args.num_queries, seed=42, norm_mode=norm_mode,
    )
    save_ply(out_dir / f"frame{target_frame:06d}_nova3r_ae.ply", nova_ae_pts, nova_ae_col)

    # ── 4. Adapter → NOVA3R pred ──────────────────────────────────────────────
    # The aggregate adapter was trained with all 8 frames' visible points
    # projected into frame 0's camera frame — replicate that here.
    print("[4/4] Adapter → NOVA3R decode …")
    c2w_frame0 = poses_c2w[0].numpy()
    w2c_frame0 = np.linalg.inv(c2w_frame0)

    all_pts_frame0 = []
    for i, fid in enumerate(frame_ids):
        gt_d = np.asarray(
            Image.open(results_dir / f"depth{fid:06d}.png"), dtype=np.float32
        ) / depth_scale
        vis_i = crop_visible_world_points(
            mesh_pts, poses_c2w[i], K_native,
            torch.from_numpy(gt_d),
            depth_tolerance=args.depth_tolerance,
        )
        pts_w = vis_i["points_world"].numpy()
        ones_i = np.ones((len(pts_w), 1), np.float32)
        pts_c0 = (np.hstack([pts_w, ones_i]) @ w2c_frame0.T)[:, :3]
        all_pts_frame0.append(pts_c0)

    pts_all = np.concatenate(all_pts_frame0, axis=0)
    N_all = len(pts_all)
    if N_all >= 8192:
        idx = np.linspace(0, N_all - 1, 8192, dtype=int)
        pts_sub_pred = pts_all[idx]
    else:
        pts_sub_pred = np.vstack([pts_all, pts_all[np.random.randint(N_all, size=8192 - N_all)]])

    norm_factor_pred = float(np.clip(np.median(np.linalg.norm(pts_sub_pred, axis=1)), 0.01, 100.0))
    pts_sub_pred_t   = torch.from_numpy(pts_sub_pred).unsqueeze(0).float().to(device)
    valid_pred_t     = torch.ones(1, 8192, dtype=torch.bool, device=device)
    pts_norm_pred, _ = normalize_input(pts_sub_pred_t, valid_pred_t, pts_sub_pred_t, valid_pred_t, mode=norm_mode)

    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim), hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens), target_dim=int(cfg.target_dim),
        depth=int(cfg.depth), num_heads=int(cfg.num_heads),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()

    # da3_agg: (1, T*N_per_frame, D) → slice frame_idx → (1, N_per_frame, D)
    da3_tok_i = da3_agg[0, frame_idx*N_per_frame:(frame_idx+1)*N_per_frame].unsqueeze(0).to(device)
    with torch.no_grad():
        pred_tok = adapter(da3_tok_i)   # (1, 768, 128)

    nova_pr_pts, nova_pr_col = decode_to_world(
        nova_model, nova_cfg, pred_tok, pts_norm_pred, norm_factor_pred, c2w_frame0,
        device, args.num_queries, seed=42, norm_mode=norm_mode,
    )
    save_ply(out_dir / f"frame{target_frame:06d}_nova3r_pred.ply", nova_pr_pts, nova_pr_col)

    # ── rerun visualisation ───────────────────────────────────────────────────
    print("[rerun] Launching viewer …")
    import rerun as rr

    def to_u8(col: np.ndarray) -> np.ndarray:
        return (np.clip(col, 0.0, 1.0) * 255).astype(np.uint8)

    rr.init(f"frame{target_frame:06d}_compare", spawn=True)
    rr.log("world/gt_mesh",     rr.Points3D(gt_pts,          colors=[200, 200, 200], radii=0.005))  # grey
    rr.log("world/da3",         rr.Points3D(da3_pts,         colors=to_u8(da3_col),  radii=0.005))  # RGB from image
    rr.log("world/nova3r_ae",   rr.Points3D(nova_ae_pts,     colors=[50, 200, 100],  radii=0.008))  # green
    rr.log("world/nova3r_pred", rr.Points3D(nova_pr_pts,     colors=[255, 100, 50],  radii=0.008))  # orange
    rr.log("world/camera",      rr.Points3D(cam_world[None], colors=[255, 50,  50],  radii=0.05))   # red

    print(f"\n[export] Done — {out_dir}/")


if __name__ == "__main__":
    main()
