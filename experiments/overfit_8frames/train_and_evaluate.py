#!/usr/bin/env python3
"""
Train DA3→NOVA3R adapter on 8-frame data AND evaluate with NOVA3R decoder.

For each stride:
  1. Re-trains the adapter (logging loss curve to wandb)
  2. Decodes z_star → GT pointcloud via NOVA3R ODE solver
  3. Decodes pred_tokens → predicted pointcloud via NOVA3R ODE solver
  4. Computes Chamfer distance (pred vs GT decoded, GT decoded vs input pts)
  5. Logs frames, input pointclouds, decoded pointclouds, and metrics to wandb

Usage:
  python train_and_evaluate.py --stride 50
  python train_and_evaluate.py --stride 150
  python train_and_evaluate.py --stride 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
from PIL import Image

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


# ── Chamfer distance ──────────────────────────────────────────────────────────

def chamfer_distance_pt(x: torch.Tensor, y: torch.Tensor) -> float:
    """Bidirectional Chamfer distance using pytorch3d. x, y: (N, 3)."""
    from pytorch3d.loss import chamfer_distance as cd3
    x = x.unsqueeze(0).float()
    y = y.unsqueeze(0).float()
    loss, _ = cd3(x, y)
    return float(loss)


# ── NOVA3R decode ─────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokens(
    nova_model,
    nova_cfg,
    tokens: torch.Tensor,      # (1, K, D) — either z_star or pred_tokens
    pts_norm: torch.Tensor,    # (1, N, 3) normalized input pts (for pointmaps arg)
    device: torch.device,
    num_queries: int = 16384,
    seed: int = 42,
) -> torch.Tensor:
    """Run NOVA3R ODE solver conditioned on `tokens`. Returns (num_queries, 3)."""
    torch.manual_seed(seed)
    B = 1
    encoder_data = {"tokens": tokens.to(device)}

    # Dummy images — AE doesn't use images but wrapper expects the arg
    images = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init = torch.rand(B, num_queries, 3, device=device) * 2 - 1

    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)

    step_size = nova_cfg.get("fm_step_size", 0.04)
    method    = nova_cfg.get("fm_sampling", "euler")

    amp_dtype_key = nova_cfg.get("amp_dtype", "bf16")
    amp_dtype     = amp_dtype_mapping.get(amp_dtype_key, torch.float32)

    T = torch.linspace(0, 1, int(1 // step_size)).to(device)

    use_amp = (device.type != "cpu")
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
        sol = solver.sample(
            time_grid        = T,
            x_init           = x_init,
            method           = method,
            step_size        = step_size,
            return_intermediates = False,
            images           = images,
            token_mask       = None,
            encoder_data     = encoder_data,
            pointmaps        = pts_norm,
        )

    pts3d = (sol[-1] if isinstance(sol, list) else sol)  # (1, N, 3)
    return pts3d[0].cpu()   # (N, 3)


# ── Pointcloud rendering ──────────────────────────────────────────────────────

def render_pts(pts: np.ndarray, color_vals: np.ndarray,
               cmap: str, title: str,
               cam_pos: np.ndarray | None = None,
               vmin=None, vmax=None, cbar_label: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    vmin = vmin if vmin is not None else np.percentile(color_vals, 2)
    vmax = vmax if vmax is not None else np.percentile(color_vals, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cols = matplotlib.colormaps[cmap](norm(color_vals))
    step = max(1, len(pts) // 20_000)
    ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1],
               c=cols[::step], s=0.6, linewidths=0, alpha=0.75)
    if cam_pos is not None:
        ax.scatter([cam_pos[0]], [cam_pos[2]], [cam_pos[1]],
                   color="red", s=80, zorder=10, marker="^", label="camera")
        ax.legend(fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.view_init(elev=25, azim=-60)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cb.set_label(cbar_label, fontsize=8)
    plt.tight_layout()
    return fig


def fig_to_wandb(fig, caption: str = "", use_wandb: bool = True):
    if not use_wandb:
        plt.close(fig)
        return None
    import wandb, io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buf).copy(), caption=caption)


# ── wandb helpers ─────────────────────────────────────────────────────────────

def init_wandb(project: str, run_name: str, cfg_dict: dict):
    import wandb
    run = wandb.init(
        project = project,
        name    = run_name,
        config  = cfg_dict,
        tags    = ["overfit", "single-scene", "proof-of-concept", "with-decoder"],
    )
    print(f"[wandb] {run.url}")
    return run


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",  type=int, required=True)
    parser.add_argument("--config",  type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--steps",   type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=16384,
                        help="Query points for NOVA3R ODE decoder")
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb",   action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    device = torch.device(args.device)
    use_wandb = args.wandb

    EXP_DIR  = REPO_ROOT / "experiments" / "overfit_8frames"
    data_dir = EXP_DIR / "data"       / f"stride_{args.stride}"
    ckpt_dir = EXP_DIR / "checkpoints"/ f"stride_{args.stride}"
    viz_dir  = EXP_DIR / "viz"        / f"stride_{args.stride}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True,  exist_ok=True)

    # ── load saved data ───────────────────────────────────────────────────────
    da3_tokens = torch.load(data_dir / "da3_tokens.pt", map_location=device, weights_only=True)
    z_star     = torch.load(data_dir / "z_star.pt",     map_location=device, weights_only=True)
    meta       = torch.load(data_dir / "meta.pt",       map_location="cpu",  weights_only=False)

    frame_ids = meta["frame_ids"]
    poses_c2w = meta["poses_c2w"]          # (T, 4, 4)
    K_native  = meta["K_native"].float()   # (3, 3)

    print(f"[train_eval stride={args.stride}] da3_tokens: {tuple(da3_tokens.shape)}")
    print(f"[train_eval stride={args.stride}] z_star    : {tuple(z_star.shape)}")

    replica_root = Path(cfg.replica_root)
    room         = str(meta["room"])
    results_dir  = replica_root / room / "results"
    mesh_ply     = replica_root / f"{room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam_params = json.load(f)["camera"]
    depth_scale = float(cam_params["scale"])

    T_frames = len(frame_ids)

    # ── build adapter ─────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
    ).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[train_eval] Adapter params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr           = float(cfg.lr),
        weight_decay = float(cfg.weight_decay),
    )

    # ── init wandb ────────────────────────────────────────────────────────────
    run_name = f"overfit_stride{args.stride}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = None
    if use_wandb:
        run = init_wandb(
            project  = cfg.wandb_project,
            run_name = run_name,
            cfg_dict = {
                "stride":         args.stride,
                "steps":          int(cfg.steps),
                "lr":             float(cfg.lr),
                "weight_decay":   float(cfg.weight_decay),
                "hidden_dim":     int(cfg.hidden_dim),
                "depth":          int(cfg.depth),
                "num_heads":      int(cfg.num_heads),
                "source_dim":     int(cfg.source_dim),
                "target_tokens":  int(cfg.target_tokens),
                "target_dim":     int(cfg.target_dim),
                "n_params":       n_params,
                "num_queries_decode": args.num_queries,
                "room":           room,
            },
        )
        import wandb

    # ── log RGB frames ────────────────────────────────────────────────────────
    print("[train_eval] Logging RGB frames …")
    colors_tab = cm.tab10(np.linspace(0, 1, T_frames))
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    frame_imgs = []
    for ax, fid, col in zip(axes.flat, frame_ids, colors_tab):
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        if use_wandb:
            frame_imgs.append(wandb.Image(img, caption=f"frame {fid:06d}"))
        ax.imshow(img)
        ax.set_title(f"frame {fid:06d}", fontsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"8 sampled frames (room0, stride={args.stride})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if use_wandb:
        grid_img = fig_to_wandb(fig, caption="RGB frames grid", use_wandb=use_wandb)
        wandb.log({"frames/grid": grid_img, "frames/individual": frame_imgs}, step=0)
    else:
        plt.close(fig)

    # ── load mesh & build per-frame visible input pointclouds ─────────────────
    print("[train_eval] Loading mesh & building per-frame input pointclouds …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    all_vis_world = []
    frame_pc_images = []
    for i, fid in enumerate(frame_ids):
        depth_img = Image.open(results_dir / f"depth{fid:06d}.png")
        depth = torch.from_numpy(np.asarray(depth_img, dtype=np.float32) / depth_scale)
        vis = crop_visible_world_points(
            points_world    = mesh_pts,
            camera_to_world = poses_c2w[i],
            intrinsics      = K_native,
            depth           = depth,
            depth_tolerance = args.depth_tolerance,
        )
        pts = vis["points_world"].numpy()
        all_vis_world.append(pts)
        print(f"  frame {fid:06d}: {len(pts):,} visible pts")

        if use_wandb:
            cam_pos = poses_c2w[i, :3, 3].numpy()
            fig = render_pts(pts, pts[:, 1], "plasma",
                             f"Frame {fid:06d} — input visible pts (height coloring)",
                             cam_pos=cam_pos, cbar_label="Y height (m)")
            frame_pc_images.append(fig_to_wandb(fig, caption=f"input PC frame {fid:06d}", use_wandb=use_wandb))

    if use_wandb:
        wandb.log({"pointclouds/input_per_frame": frame_pc_images}, step=0)

    # Combined input pointcloud (colored by frame index)
    tab10 = matplotlib.colormaps["tab10"]
    all_pts_cat = np.concatenate(all_vis_world, axis=0)
    all_fidx    = np.concatenate([np.full(len(p), i) for i, p in enumerate(all_vis_world)])
    fig2 = plt.figure(figsize=(11, 8))
    ax2  = fig2.add_subplot(111, projection="3d")
    step2 = max(1, len(all_pts_cat) // 40_000)
    point_colors = tab10(all_fidx / max(T_frames - 1, 1))
    ax2.scatter(all_pts_cat[::step2, 0], all_pts_cat[::step2, 2],
                all_pts_cat[::step2, 1], c=point_colors[::step2], s=0.4, alpha=0.6)
    for i, fid in enumerate(frame_ids):
        cam = poses_c2w[i, :3, 3].numpy()
        ax2.scatter(cam[0], cam[2], cam[1], color=tab10(i / max(T_frames-1,1)),
                    s=120, marker="^", zorder=10, edgecolors="k", linewidths=0.5)
        ax2.text(cam[0], cam[2], cam[1]+0.05, str(fid), fontsize=7,
                 color=tab10(i / max(T_frames-1,1)), ha="center")
    ax2.set_title(f"All visible pts — colored by frame (stride={args.stride})",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_zlabel("Y")
    ax2.view_init(elev=20, azim=-50)
    if use_wandb:
        wandb.log({"pointclouds/input_all_frames": fig_to_wandb(fig2, "all input PCs", use_wandb=use_wandb)}, step=0)
    else:
        plt.close(fig2)

    # Build normalized input pts (in first-camera frame, same as NOVA3R encoder input)
    # We need these for the decoder's pointmaps argument
    first_pose = poses_c2w[0]  # (4, 4)
    w2c_first  = torch.linalg.inv(first_pose.float())

    def world_to_first_cam(pts_w: np.ndarray) -> torch.Tensor:
        pts_t = torch.from_numpy(pts_w.astype(np.float32))
        h = torch.ones(len(pts_t), 1)
        pts_h = torch.cat([pts_t, h], dim=1)  # (N, 4)
        return (pts_h @ w2c_first.T)[:, :3]    # (N, 3)

    pts_first_cam_list = [world_to_first_cam(p) for p in all_vis_world]
    pts_first_cam_all  = torch.cat(pts_first_cam_list, dim=0)  # (N_total, 3)
    N_total = len(pts_first_cam_all)
    # Uniform subsample to 8192
    if N_total > 8192:
        idx = torch.linspace(0, N_total - 1, 8192).long()
        pts_sub = pts_first_cam_all[idx]
    else:
        pts_sub = pts_first_cam_all
    pts_sub = pts_sub.unsqueeze(0)  # (1, N, 3)

    valid = torch.ones(1, pts_sub.shape[1], dtype=torch.bool)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"[train_eval] Training for {int(cfg.steps)} steps …")
    adapter.train()
    t0 = time.time()
    loss_history = []

    for step in range(1, int(cfg.steps) + 1):
        pred = adapter(da3_tokens)
        loss = F.mse_loss(pred, z_star)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % int(cfg.log_every) == 0 or step == 1 or step == int(cfg.steps):
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{cfg.steps}  loss={loss_val:.6f}  t={elapsed:.1f}s")
            if use_wandb:
                wandb.log({"train/loss": loss_val, "train/step": step}, step=step)

    final_loss = loss_val
    print(f"[train_eval] Training done. Final loss: {final_loss:.6f}")

    # Save checkpoint
    ckpt_path = ckpt_dir / "overfit_final.pt"
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "config": OmegaConf.to_container(cfg),
        "final_loss": final_loss,
        "steps": int(cfg.steps),
    }, ckpt_path)
    print(f"[train_eval] Checkpoint saved: {ckpt_path}")
    if use_wandb:
        wandb.log({"train/final_loss": final_loss}, step=int(cfg.steps))

    # ── get predicted tokens ──────────────────────────────────────────────────
    adapter.eval()
    with torch.no_grad():
        pred_tokens = adapter(da3_tokens)   # (1, 768, 128)
    print(f"[train_eval] pred_tokens: {tuple(pred_tokens.shape)}")
    print(f"[train_eval] MSE(pred, z_star) = {F.mse_loss(pred_tokens, z_star).item():.6f}")

    # ── load NOVA3R AE for decoding ───────────────────────────────────────────
    nova3r_ckpt = Path(str(cfg.nova3r_ckpt))
    print(f"[train_eval] Loading NOVA3R AE from {nova3r_ckpt} …")
    nova_model, nova_cfg = load_nova3r_model(str(nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)

    # Normalize input pts for decoder's `pointmaps` argument
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    pts_norm, _ = normalize_input(
        pts_sub.to(device), valid.to(device),
        pts_sub.to(device), valid.to(device),
        mode=norm_mode,
    )
    print(f"[train_eval] pts_norm: {tuple(pts_norm.shape)}, norm_mode={norm_mode}")

    # ── decode GT z_star ──────────────────────────────────────────────────────
    print(f"[train_eval] Decoding GT z_star (ODE, {args.num_queries} query pts) …")
    t_dec = time.time()
    gt_decoded = decode_tokens(nova_model, nova_cfg, z_star, pts_norm, device,
                               num_queries=args.num_queries, seed=42)
    print(f"  done in {time.time()-t_dec:.1f}s  →  {tuple(gt_decoded.shape)}")

    # ── decode predicted tokens ───────────────────────────────────────────────
    print(f"[train_eval] Decoding predicted tokens (ODE, {args.num_queries} query pts) …")
    t_dec = time.time()
    pred_decoded = decode_tokens(nova_model, nova_cfg, pred_tokens, pts_norm, device,
                                 num_queries=args.num_queries, seed=42)
    print(f"  done in {time.time()-t_dec:.1f}s  →  {tuple(pred_decoded.shape)}")

    # ── Chamfer distances ─────────────────────────────────────────────────────
    print("[train_eval] Computing Chamfer distances …")

    # 1. Between pred_decoded and gt_decoded (adapter quality in decoded space)
    cd_pred_vs_gt = chamfer_distance_pt(pred_decoded, gt_decoded)

    # 2. Between gt_decoded and original normalized input pts (AE quality baseline)
    pts_norm_np = pts_norm[0].cpu().numpy()
    gt_dec_np   = gt_decoded.numpy()
    cd_ae_recon = chamfer_distance_pt(
        torch.from_numpy(gt_dec_np),
        torch.from_numpy(pts_norm_np),
    )

    # 3. Between pred_decoded and original normalized input pts
    cd_pred_vs_input = chamfer_distance_pt(
        pred_decoded,
        torch.from_numpy(pts_norm_np),
    )

    print(f"  CD(pred_decoded  vs  gt_decoded)     = {cd_pred_vs_gt:.6f}")
    print(f"  CD(gt_decoded    vs  input_pts_norm) = {cd_ae_recon:.6f}  [AE baseline]")
    print(f"  CD(pred_decoded  vs  input_pts_norm) = {cd_pred_vs_input:.6f}")

    if use_wandb:
        wandb.log({
            "eval/cd_pred_vs_gt_decoded":   cd_pred_vs_gt,
            "eval/cd_ae_recon_vs_input":    cd_ae_recon,
            "eval/cd_pred_decoded_vs_input": cd_pred_vs_input,
            "eval/final_mse_loss":          final_loss,
        }, step=int(cfg.steps))

    # ── visualize decoded pointclouds ─────────────────────────────────────────
    print("[train_eval] Rendering decoded pointclouds …")

    # GT decoded
    fig_gt = render_pts(gt_dec_np, gt_dec_np[:, 1], "viridis",
                        f"GT decoded (z_star → ODE) — stride={args.stride}\n"
                        f"CD vs input: {cd_ae_recon:.4f}",
                        cbar_label="Y height (normalized)")

    # Pred decoded
    pred_dec_np = pred_decoded.numpy()
    fig_pred = render_pts(pred_dec_np, pred_dec_np[:, 1], "plasma",
                          f"Pred decoded (pred_tokens → ODE) — stride={args.stride}\n"
                          f"CD vs GT decoded: {cd_pred_vs_gt:.4f}",
                          cbar_label="Y height (normalized)")

    # Input pts (normalized) for comparison
    fig_in = render_pts(pts_norm_np, pts_norm_np[:, 1], "cividis",
                        f"Input pts (normalized, first-cam frame) — stride={args.stride}",
                        cbar_label="Y height (normalized)")

    if use_wandb:
        wandb.log({
            "pointclouds/input_normalized":  fig_to_wandb(fig_in,   f"Input normalized pts (stride={args.stride})", use_wandb=use_wandb),
            "pointclouds/gt_decoded":        fig_to_wandb(fig_gt,   f"GT decoded pointcloud (stride={args.stride})", use_wandb=use_wandb),
            "pointclouds/pred_decoded":      fig_to_wandb(fig_pred, f"Pred decoded pointcloud (stride={args.stride})", use_wandb=use_wandb),
        }, step=int(cfg.steps))
    else:
        plt.close(fig_gt); plt.close(fig_pred); plt.close(fig_in)

    # Side-by-side comparison figure
    fig_comp, axes_comp = plt.subplots(1, 3, figsize=(27, 8),
                                       subplot_kw={"projection": "3d"})
    datasets = [
        (pts_norm_np,  "cividis", "Input pts (normalized)"),
        (gt_dec_np,    "viridis", f"GT decoded\nCD_AE={cd_ae_recon:.4f}"),
        (pred_dec_np,  "plasma",  f"Pred decoded\nCD_pred={cd_pred_vs_gt:.4f}"),
    ]
    for ax_c, (pts_c, cmap_c, title_c) in zip(axes_comp, datasets):
        vmin_c = np.percentile(pts_c[:, 1], 2)
        vmax_c = np.percentile(pts_c[:, 1], 98)
        norm_c = Normalize(vmin=vmin_c, vmax=vmax_c)
        cols_c = matplotlib.colormaps[cmap_c](norm_c(pts_c[:, 1]))
        step_c = max(1, len(pts_c) // 15_000)
        ax_c.scatter(pts_c[::step_c, 0], pts_c[::step_c, 2], pts_c[::step_c, 1],
                     c=cols_c[::step_c], s=0.8, alpha=0.75)
        ax_c.set_title(title_c, fontsize=10, fontweight="bold")
        ax_c.set_xlabel("X"); ax_c.set_ylabel("Z"); ax_c.set_zlabel("Y")
        ax_c.view_init(elev=25, azim=-60)
        ax_c.tick_params(labelsize=7)

    fig_comp.suptitle(
        f"Stride={args.stride} | MSE={final_loss:.4e} | "
        f"CD(pred↔gt)={cd_pred_vs_gt:.4f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if use_wandb:
        wandb.log({"pointclouds/comparison": fig_to_wandb(fig_comp, "side-by-side comparison", use_wandb=use_wandb)},
                  step=int(cfg.steps))
    else:
        plt.close(fig_comp)

    # ── log wandb.Object3D pointclouds ────────────────────────────────────────
    if use_wandb:
        print("[train_eval] Logging 3D point cloud objects to wandb …")
        wandb.log({
            "3d/input_pts": wandb.Object3D(
                {"type": "lidar/beta",
                 "points": pts_norm_np[:, [0, 2, 1]].astype(np.float32)}),
            "3d/gt_decoded": wandb.Object3D(
                {"type": "lidar/beta",
                 "points": gt_dec_np[:, [0, 2, 1]].astype(np.float32)}),
            "3d/pred_decoded": wandb.Object3D(
                {"type": "lidar/beta",
                 "points": pred_dec_np[:, [0, 2, 1]].astype(np.float32)}),
        }, step=int(cfg.steps))

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STRIDE {args.stride} RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Training steps:                {int(cfg.steps)}")
    print(f"  Final MSE loss:                {final_loss:.6f}")
    print(f"  CD(pred_decoded vs gt_decoded):{cd_pred_vs_gt:.6f}")
    print(f"  CD(gt_decoded vs input_norm):  {cd_ae_recon:.6f}  [AE baseline]")
    print(f"  CD(pred_decoded vs input_norm):{cd_pred_vs_input:.6f}")
    if use_wandb:
        print(f"  W&B run: {run.url}")
        wandb.finish()
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
