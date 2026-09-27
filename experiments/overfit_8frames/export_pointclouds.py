#!/usr/bin/env python3
"""
Export per-frame pointclouds – all four sources in world frame.

For each stride × each frame:
  frame_XXXXXX_gt_mesh.ply      GT visible mesh pts (world, height-colored)
  frame_XXXXXX_da3.ply          DA3 depth backprojected (world, RGB from image)
  frame_XXXXXX_nova3r_gt.ply    NOVA3R decode of GT z_star (world, height-colored)
  frame_XXXXXX_nova3r_pred.ply  NOVA3R decode of pred tokens (world, height-colored)

All four are in the same world frame → directly overlay-able in any 3D viewer.

Usage:
  python export_pointclouds.py --stride 20
  python export_pointclouds.py --stride 50
  python export_pointclouds.py --stride 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
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

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.inference import normalize_input, amp_dtype_mapping  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from vc3r.replica import crop_visible_world_points        # noqa: E402
from depth_anything_3.cfg import create_object                             # noqa: E402


# ── PLY helpers ───────────────────────────────────────────────────────────────

def save_ply(path: Path, pts: np.ndarray, colors: np.ndarray):
    """Save colored PLY. pts: (N,3) float32, colors: (N,3) float32 [0,1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def height_cmap(pts: np.ndarray, cmap="plasma") -> np.ndarray:
    y = pts[:, 1]
    norm = Normalize(np.percentile(y, 2), np.percentile(y, 98))
    return plt.get_cmap(cmap)(norm(y))[:, :3].astype(np.float32)


# ── DA3 model loading ─────────────────────────────────────────────────────────

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
    transform   = torch.linalg.inv(w2c[:, :1])
    normalized  = w2c @ transform
    c2ws        = torch.linalg.inv(normalized)
    median_dist = float(c2ws[..., :3, 3].norm(dim=-1).median().clamp(min=1e-1))
    normalized[..., :3, 3] /= median_dist
    return normalized, median_dist


# ── depth → colored world-frame pointcloud ────────────────────────────────────

def depth_to_world_rgb(depth: np.ndarray, rgb: np.ndarray,
                        K: np.ndarray, c2w: np.ndarray,
                        max_pts: int = 60_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth map to world frame, sampling RGB colors.
    depth: (H, W) meters
    rgb:   (H, W, 3) float32 [0,1]  at same resolution as depth
    Returns pts_world (N,3), colors (N,3) float32 [0,1]
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    mask = np.isfinite(depth) & (depth > 0.1)
    d  = depth[mask]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x  = (uu[mask] - cx) / fx * d
    y  = (vv[mask] - cy) / fy * d
    z  = d
    cols = rgb[mask]           # (N, 3)
    ones = np.ones(len(d), dtype=np.float32)
    pts_cam = np.stack([x, y, z, ones], axis=1)   # (N, 4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]        # (N, 3)
    if len(pts_world) > max_pts:
        idx = np.random.choice(len(pts_world), max_pts, replace=False)
        pts_world = pts_world[idx]; cols = cols[idx]
    return pts_world.astype(np.float32), cols.astype(np.float32)


def align_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    mask = (gt > 0.1) & np.isfinite(gt) & (pred > 1e-6)
    d, g = pred[mask].ravel(), gt[mask].ravel()
    return float(max((d * g).sum() / (d * d + 1e-12).sum(), 0.01))


# ── NOVA3R: normalize → decode → un-normalize → world ────────────────────────

def compute_nova3r_norm_factor(pts_cam: np.ndarray) -> float:
    """
    Replicate normalize_input(mode='median_3') scale factor from camera-frame pts.
    pts_cam: (N, 3)  in camera frame (origin = camera, Z = depth direction)
    """
    dists = np.linalg.norm(pts_cam, axis=1)   # distances from camera origin
    norm_factor = float(np.median(dists))
    return float(np.clip(norm_factor, 0.01, 100.0))


@torch.no_grad()
def decode_to_world(nova_model, nova_cfg,
                    tokens: torch.Tensor,          # (1, 768, 128)
                    pts_norm: torch.Tensor,         # (1, 8192, 3)  already normalized
                    norm_factor: float,
                    c2w: np.ndarray,               # (4, 4) world transform
                    device: torch.device,
                    num_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode NOVA3R tokens → un-normalize → world frame.
    Returns pts_world (N,3), colors (N,3) from height colormap.
    target_median=3 (from norm_mode='median_3')
    """
    torch.manual_seed(seed)
    B = 1
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(B, num_queries, 3, device=device) * 2 - 1
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
    pts_norm_dec = (sol[-1] if isinstance(sol, list) else sol)[0].cpu().numpy()  # (N, 3)

    # Invert normalize_input(mode='median_3'):  pts_norm = pts_cam / norm_factor * 3
    # → pts_cam = pts_norm_dec / 3 * norm_factor
    pts_cam = pts_norm_dec / 3.0 * norm_factor   # (N, 3)  camera frame

    # Camera frame → world frame
    ones = np.ones((len(pts_cam), 1), dtype=np.float32)
    pts_world = (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3]

    colors = height_cmap(pts_world, cmap="plasma")
    return pts_world.astype(np.float32), colors


# ── matplotlib 3D scatter (high quality) ─────────────────────────────────────

def scatter3d(ax, pts: np.ndarray, colors: np.ndarray,
              max_pts: int = 25_000, s: float = 2.5, alpha: float = 0.9):
    step = max(1, len(pts) // max_pts)
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
               c=colors[::step], s=s, linewidths=0, alpha=alpha, depthshade=False)


def set_equal_aspect(ax, pts_list):
    """Set equal axis limits across all point sets."""
    all_pts = np.concatenate(pts_list, axis=0)
    mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2
    half   = (maxs - mins).max() / 2 * 0.6
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)


def render_comparison(frame_id: int, stride: int,
                       gt_pts: np.ndarray, gt_col: np.ndarray,
                       da3_pts: np.ndarray, da3_col: np.ndarray,
                       nova_gt_pts: np.ndarray, nova_gt_col: np.ndarray,
                       nova_pr_pts: np.ndarray, nova_pr_col: np.ndarray,
                       cam_pos: np.ndarray) -> plt.Figure:
    """4-panel + 1 overlay figure, all in world frame."""
    fig = plt.figure(figsize=(40, 9))

    titles = [
        "GT mesh (world, height)",
        "DA3 depth (world, RGB)",
        "NOVA3R GT decoded (world, height)",
        "NOVA3R pred decoded (world, height)",
        "Overlay: GT mesh + DA3 + NOVA3R pred",
    ]
    datasets = [
        (gt_pts,      gt_col),
        (da3_pts,     da3_col),
        (nova_gt_pts, nova_gt_col),
        (nova_pr_pts, nova_pr_col),
        None,   # overlay handled separately
    ]
    elev, azim = 20, -55

    for col_idx in range(5):
        ax = fig.add_subplot(1, 5, col_idx+1, projection="3d")

        if col_idx < 4:
            pts, col = datasets[col_idx]
            scatter3d(ax, pts, col, s=3.0)
        else:
            # Overlay: GT mesh (grey), DA3 (original RGB), NOVA3R pred (red/orange)
            scatter3d(ax, gt_pts,     np.full_like(gt_col, 0.65),  s=1.5, alpha=0.5)
            scatter3d(ax, da3_pts,    da3_col,                      s=1.5, alpha=0.7)
            scatter3d(ax, nova_pr_pts, nova_pr_col,                 s=2.5, alpha=0.9)

        # Camera marker
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                   color="red", s=120, marker="^", zorder=10,
                   edgecolors="k", linewidths=0.5)

        all_pts = [gt_pts, da3_pts, nova_gt_pts, nova_pr_pts]
        set_equal_aspect(ax, all_pts)
        ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_zlabel("Z (m)", fontsize=7)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)
        ax.set_title(titles[col_idx], fontsize=9, fontweight="bold", pad=4)

    fig.suptitle(f"Frame {frame_id:06d} | stride={stride} | all in world frame",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_to_wandb(fig, caption=""):
    import wandb, io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buf).copy(), caption=caption)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",       type=int, required=True)
    parser.add_argument("--config",       type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--num-queries",  type=int, default=8192)
    parser.add_argument("--max-pts",      type=int, default=60_000)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)

    EXP_DIR   = REPO_ROOT / "experiments" / "overfit_8frames"
    data_dir  = EXP_DIR / "data"        / f"stride_{args.stride}"
    ckpt_path = EXP_DIR / "checkpoints" / f"stride_{args.stride}_perframe" / "overfit_perframe_final.pt"
    ply_dir   = EXP_DIR / "ply"         / f"stride_{args.stride}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    # ── load metadata ─────────────────────────────────────────────────────────
    da3_agg    = torch.load(data_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    meta       = torch.load(data_dir / "meta.pt",       map_location="cpu", weights_only=False)
    frame_ids  = meta["frame_ids"]
    poses_c2w  = meta["poses_c2w"].float()   # (T, 4, 4)
    K_native   = meta["K_native"].float()    # (3, 3)
    T_frames   = len(frame_ids)
    N_per_frame = da3_agg.shape[1] // T_frames

    replica_root = Path("/storage/local/Replica")
    room         = str(meta["room"])
    results_dir  = replica_root / room / "results"
    mesh_ply     = replica_root / f"{room}_mesh.ply"
    with (replica_root / "cam_params.json").open() as f:
        cam_params = json.load(f)["camera"]
    depth_scale = float(cam_params["scale"])
    W_nat, H_nat = int(cam_params["w"]), int(cam_params["h"])

    W_da3, H_da3 = int(cfg.image_width), int(cfg.image_height)
    sx, sy = W_da3 / W_nat, H_da3 / H_nat
    K_da3 = K_native.numpy().copy()
    K_da3[0] *= sx; K_da3[1] *= sy

    print(f"[export stride={args.stride}] T={T_frames}, N/frame={N_per_frame}")

    # ── load mesh ─────────────────────────────────────────────────────────────
    print("[export] Sampling mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    # ── load DA3 and run depth inference ─────────────────────────────────────
    print("[export] Loading DA3 …")
    da3_model = load_da3(str(cfg.da3_model), device)

    images_list = []
    rgb_list    = []   # (H_da3, W_da3, 3) float32 [0,1] per frame
    for fid in frame_ids:
        img_pil = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img_da3 = img_pil.resize((W_da3, H_da3), Image.BILINEAR)
        rgb_np  = np.array(img_da3, dtype=np.float32) / 255.0
        rgb_list.append(rgb_np)
        images_list.append(torch.from_numpy(rgb_np).permute(2, 0, 1))

    images      = torch.stack(images_list).unsqueeze(0).to(device)   # (1,T,3,H,W)
    images_norm = imagenet_norm(images)

    w2c_all = torch.linalg.inv(poses_c2w).unsqueeze(0)
    extr_norm, median_dist = normalize_extrinsics(w2c_all)
    K_batch = K_native.unsqueeze(0).unsqueeze(0).expand(1, T_frames, -1, -1).to(device)

    print("[export] Running DA3 depth inference …")
    with torch.no_grad():
        amp_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dt,
                            enabled=(device.type == "cuda")):
            cam_tok = da3_model.cam_enc(extr_norm.to(device), K_batch, images_norm.shape[-2:])
            feats, _ = da3_model.backbone(images_norm, cam_token=cam_tok)
        with torch.autocast(device_type=device.type, enabled=False):
            da3_out = da3_model._process_depth_head(feats, H_da3, W_da3)
    da3_depth_all = da3_out.depth[0].cpu().numpy()   # (T, H, W) normalized scale

    # ── load NOVA3R AE ────────────────────────────────────────────────────────
    print("[export] Loading NOVA3R AE …")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── load per-frame adapter ────────────────────────────────────────────────
    print(f"[export] Loading adapter checkpoint …")
    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim), hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens), target_dim=int(cfg.target_dim),
        depth=int(cfg.depth), num_heads=int(cfg.num_heads),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()

    da3_batch = da3_agg.view(T_frames, N_per_frame, -1).to(device)
    with torch.no_grad():
        pred_tok_batch = adapter(da3_batch)   # (T, 768, 128)

    # ── wandb ─────────────────────────────────────────────────────────────────
    import wandb
    run = wandb.init(
        project = cfg.wandb_project,
        name    = f"export_v2_stride{args.stride}_{time.strftime('%Y%m%d_%H%M%S')}",
        config  = {"stride": args.stride, "num_queries": args.num_queries,
                   "max_pts": args.max_pts, "room": room, "version": 2},
        tags    = ["export", "world-frame", "rgb-colored", "per-frame"],
    )
    print(f"[wandb] {run.url}")

    wb_imgs = {"gt_mesh": [], "da3": [], "nova3r_gt": [], "nova3r_pred": [], "comparison": []}
    wb_3d   = {"gt_mesh": [], "da3": [], "nova3r_gt": [], "nova3r_pred": []}

    # ── per-frame loop ────────────────────────────────────────────────────────
    for i, fid in enumerate(frame_ids):
        print(f"\n── frame {fid:06d} ─────────────────────────────────────────────")
        c2w_np = poses_c2w[i].numpy()
        w2c_np = np.linalg.inv(c2w_np)
        prefix = ply_dir / f"frame{fid:06d}"

        # GT depth (native res) for scale alignment
        gt_depth_nat = np.asarray(
            Image.open(results_dir / f"depth{fid:06d}.png"), dtype=np.float32
        ) / depth_scale
        gt_depth_da3 = np.array(
            Image.fromarray(gt_depth_nat).resize((W_da3, H_da3), Image.NEAREST)
        )

        # ── GT visible mesh (world, height-colored) ───────────────────────────
        vis = crop_visible_world_points(
            mesh_pts, poses_c2w[i], K_native,
            torch.from_numpy(gt_depth_nat),
            depth_tolerance=args.depth_tolerance,
        )
        gt_pts = vis["points_world"].numpy()
        if len(gt_pts) > args.max_pts:
            gt_pts = gt_pts[np.random.choice(len(gt_pts), args.max_pts, replace=False)]
        gt_col = height_cmap(gt_pts, "YlOrRd")
        save_ply(Path(f"{prefix}_gt_mesh.ply"), gt_pts, gt_col)
        print(f"  GT mesh:      {len(gt_pts):,} pts")

        # ── DA3 depth → world, RGB colored ────────────────────────────────────
        da3_raw   = da3_depth_all[i]
        scale     = align_scale(da3_raw, gt_depth_da3)
        da3_depth = da3_raw * scale
        da3_pts, da3_col = depth_to_world_rgb(
            da3_depth, rgb_list[i], K_da3, c2w_np, args.max_pts
        )
        save_ply(Path(f"{prefix}_da3.ply"), da3_pts, da3_col)
        print(f"  DA3 (scale={scale:.3f}): {len(da3_pts):,} pts")

        # ── NOVA3R: encode this frame's visible pts, then decode ──────────────
        # Camera-frame input pts (frame i's own cam)
        ones     = np.ones((len(gt_pts), 1), dtype=np.float32)
        pts_cam  = (np.hstack([gt_pts, ones]) @ w2c_np.T)[:, :3]   # (N,3) cam frame
        n = len(pts_cam)
        if n >= 8192:
            idx = np.linspace(0, n-1, 8192, dtype=int)
            pts_sub = pts_cam[idx]
        else:
            pts_sub = np.vstack([pts_cam, pts_cam[np.random.randint(n, size=8192-n)]])

        # Compute norm_factor BEFORE normalizing (to invert later)
        norm_factor = compute_nova3r_norm_factor(pts_sub)

        pts_sub_t = torch.from_numpy(pts_sub).unsqueeze(0).float().to(device)
        valid_t   = torch.ones(1, 8192, dtype=torch.bool, device=device)
        pts_norm_t, _ = normalize_input(pts_sub_t, valid_t, pts_sub_t, valid_t, mode="median_3")

        with torch.no_grad():
            z_i = nova_model._encode(pointmaps=pts_norm_t, test=True)["tokens"].float()

        nova_gt_pts, nova_gt_col = decode_to_world(
            nova_model, nova_cfg, z_i, pts_norm_t,
            norm_factor, c2w_np, device, args.num_queries, seed=42+i,
        )
        save_ply(Path(f"{prefix}_nova3r_gt.ply"), nova_gt_pts, nova_gt_col)
        print(f"  NOVA3R GT:    {len(nova_gt_pts):,} pts (norm_factor={norm_factor:.3f})")

        nova_pr_pts, nova_pr_col = decode_to_world(
            nova_model, nova_cfg, pred_tok_batch[i:i+1], pts_norm_t,
            norm_factor, c2w_np, device, args.num_queries, seed=42+i,
        )
        save_ply(Path(f"{prefix}_nova3r_pred.ply"), nova_pr_pts, nova_pr_col)
        print(f"  NOVA3R pred:  {len(nova_pr_pts):,} pts")

        # ── Comparison figure (world frame, uniform scale) ────────────────────
        cam_world = poses_c2w[i, :3, 3].numpy()
        fig = render_comparison(
            fid, args.stride,
            gt_pts, gt_col,
            da3_pts, da3_col,
            nova_gt_pts, nova_gt_col,
            nova_pr_pts, nova_pr_col,
            cam_world,
        )
        fig.savefig(str(ply_dir / f"frame{fid:06d}_comparison.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

        # wandb images (4-way comparison)
        fig2 = render_comparison(
            fid, args.stride,
            gt_pts, gt_col, da3_pts, da3_col,
            nova_gt_pts, nova_gt_col, nova_pr_pts, nova_pr_col, cam_world,
        )
        wb_imgs["comparison"].append(fig_to_wandb(fig2, f"comparison frame {fid:06d}"))

        # Individual renders
        for key, pts_r, col_r in [
            ("gt_mesh",      gt_pts,      gt_col),
            ("da3",          da3_pts,     da3_col),
            ("nova3r_gt",    nova_gt_pts, nova_gt_col),
            ("nova3r_pred",  nova_pr_pts, nova_pr_col),
        ]:
            fig3 = plt.figure(figsize=(9, 7))
            ax3  = fig3.add_subplot(111, projection="3d")
            scatter3d(ax3, pts_r, col_r, s=3.0)
            ax3.scatter([cam_world[0]], [cam_world[1]], [cam_world[2]],
                        color="red", s=120, marker="^", zorder=10)
            set_equal_aspect(ax3, [gt_pts, da3_pts, nova_gt_pts, nova_pr_pts])
            ax3.view_init(elev=20, azim=-55)
            ax3.set_title(f"{key}  frame {fid:06d}", fontsize=10, fontweight="bold")
            ax3.tick_params(labelsize=7)
            plt.tight_layout()
            wb_imgs[key].append(fig_to_wandb(fig3, f"{key} frame {fid:06d}"))

            wb_3d[key].append(wandb.Object3D(
                {"type": "lidar/beta", "points": pts_r[:, [0, 2, 1]]}))

    # ── wandb log ─────────────────────────────────────────────────────────────
    print("\n[export] Logging to wandb …")
    wandb.log({
        "pointclouds/comparison": wb_imgs["comparison"],
        "pointclouds/gt_mesh":    wb_imgs["gt_mesh"],
        "pointclouds/da3":        wb_imgs["da3"],
        "pointclouds/nova3r_gt":  wb_imgs["nova3r_gt"],
        "pointclouds/nova3r_pred":wb_imgs["nova3r_pred"],
        "3d/gt_mesh":    wb_3d["gt_mesh"],
        "3d/da3":        wb_3d["da3"],
        "3d/nova3r_gt":  wb_3d["nova3r_gt"],
        "3d/nova3r_pred":wb_3d["nova3r_pred"],
    })

    print(f"\n[export] PLY files saved to: {ply_dir}/")
    for p in sorted(ply_dir.glob("*.ply")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

    print(f"\n[wandb] {run.url}")
    wandb.finish()


# ── helper (module-level for clarity) ────────────────────────────────────────

def compute_nova3r_norm_factor(pts_cam: np.ndarray) -> float:
    dists = np.linalg.norm(pts_cam, axis=1)
    return float(np.clip(np.median(dists), 0.01, 100.0))


if __name__ == "__main__":
    main()
