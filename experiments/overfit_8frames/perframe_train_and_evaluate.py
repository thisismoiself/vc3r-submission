#!/usr/bin/env python3
"""
Per-frame DA3→NOVA3R adapter training and evaluation.

For each stride, for each of the 8 frames independently:
  - DA3 tokens: split existing da3_tokens.pt by frame (256 tokens/frame)
  - NOVA3R z_star: re-encode each frame's visible pts in its own camera frame
  - Adapter: trained on (B=8) per-frame pairs per step
  - Eval: decode z_star_i and pred_i per frame, compute per-frame Chamfer
  - Also aggregated stats across frames

Usage:
  python perframe_train_and_evaluate.py --stride 20
  python perframe_train_and_evaluate.py --stride 50
  python perframe_train_and_evaluate.py --stride 150
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


# ── utilities ─────────────────────────────────────────────────────────────────

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Bidirectional Chamfer (pytorch3d). x, y: (N, 3)."""
    from pytorch3d.loss import chamfer_distance as cd3
    loss, _ = cd3(x.unsqueeze(0).float(), y.unsqueeze(0).float())
    return float(loss)


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens: torch.Tensor,
                  pts_norm: torch.Tensor, device: torch.device,
                  num_queries: int = 8192, seed: int = 42) -> torch.Tensor:
    """ODE decode tokens → (num_queries, 3) in normalized coords."""
    torch.manual_seed(seed)
    B = tokens.shape[0]
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(B, num_queries, 3, device=device) * 2 - 1

    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_key = nova_cfg.get("amp_dtype", "bf16")
    amp_dt  = amp_dtype_mapping.get(amp_key, torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)

    use_amp = device.type != "cpu"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm,
        )
    pts3d = sol[-1] if isinstance(sol, list) else sol   # (B, N, 3)
    return pts3d.cpu()                                   # (B, N, 3)


def render_pts(pts: np.ndarray, color_vals: np.ndarray,
               cmap: str, title: str, cam_pos=None,
               vmin=None, vmax=None, cbar_label: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    vn  = vmin if vmin is not None else np.percentile(color_vals, 2)
    vx  = vmax if vmax is not None else np.percentile(color_vals, 98)
    norm = Normalize(vmin=vn, vmax=vx)
    cols = matplotlib.colormaps[cmap](norm(color_vals))
    step = max(1, len(pts) // 15_000)
    ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1],
               c=cols[::step], s=0.8, linewidths=0, alpha=0.75)
    if cam_pos is not None:
        ax.scatter([cam_pos[0]], [cam_pos[2]], [cam_pos[1]],
                   color="red", s=80, zorder=10, marker="^", label="camera")
        ax.legend(fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.view_init(elev=25, azim=-60)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1).set_label(cbar_label, fontsize=8)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",      type=int, required=True)
    parser.add_argument("--config",      type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--steps",       type=int,   default=None)
    parser.add_argument("--num-queries", type=int,   default=8192)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb",       action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    device = torch.device(args.device)
    use_wandb = args.wandb

    EXP_DIR  = REPO_ROOT / "experiments" / "overfit_8frames"
    data_dir = EXP_DIR / "data"        / f"stride_{args.stride}"
    ckpt_dir = EXP_DIR / "checkpoints" / f"stride_{args.stride}_perframe"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── load existing aggregated data ─────────────────────────────────────────
    da3_agg = torch.load(data_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    # da3_agg: (1, T*N_per_frame, D)
    meta       = torch.load(data_dir / "meta.pt", map_location="cpu", weights_only=False)
    frame_ids  = meta["frame_ids"]
    poses_c2w  = meta["poses_c2w"].float()   # (T, 4, 4)
    K_native   = meta["K_native"].float()    # (3, 3)
    T_frames   = len(frame_ids)
    N_per_frame = da3_agg.shape[1] // T_frames   # tokens per frame
    D_da3       = da3_agg.shape[2]

    # Split: (T, 1, N_per_frame, D)
    da3_perframe = da3_agg.view(T_frames, N_per_frame, D_da3).unsqueeze(1)
    # da3_perframe[i]: (1, N_per_frame, D)  for frame i
    print(f"[perframe stride={args.stride}] T={T_frames}, N/frame={N_per_frame}, D={D_da3}")

    # ── replica paths ─────────────────────────────────────────────────────────
    replica_root = Path(cfg.replica_root)
    room         = str(meta["room"])
    results_dir  = replica_root / room / "results"
    mesh_ply     = replica_root / f"{room}_mesh.ply"
    with (replica_root / "cam_params.json").open() as f:
        depth_scale = float(json.load(f)["camera"]["scale"])

    # ── load NOVA3R AE ────────────────────────────────────────────────────────
    nova3r_ckpt = Path(str(cfg.nova3r_ckpt))
    print(f"[perframe] Loading NOVA3R AE …")
    nova_model, nova_cfg = load_nova3r_model(str(nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)
    norm_mode = nova_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    # ── load mesh ─────────────────────────────────────────────────────────────
    print("[perframe] Sampling mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    # ── per-frame visible points & NOVA3R encoding ────────────────────────────
    print("[perframe] Building per-frame visible pointclouds & z_star …")
    vis_pts_world_list  = []   # list of (N_i, 3) numpy
    vis_pts_cam_list    = []   # list of (1, 8192, 3) tensor — each frame's own camera
    vis_pts_norm_list   = []   # normalized, on device
    z_star_list         = []   # (1, 768, 128) per frame

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
        pts_world = vis["points_world"]          # (N_i, 3) world coords

        # Project to this frame's OWN camera frame
        w2c_i = torch.linalg.inv(poses_c2w[i])
        ones  = torch.ones(len(pts_world), 1)
        pts_cam_i = (torch.cat([pts_world, ones], dim=1) @ w2c_i.T)[:, :3]  # (N_i, 3)

        # Subsample to 8192
        n = len(pts_cam_i)
        if n >= 8192:
            idx = torch.linspace(0, n - 1, 8192).long()
            pts_sub = pts_cam_i[idx]
        else:
            pad = torch.randint(n, (8192 - n,))
            pts_sub = torch.cat([pts_cam_i, pts_cam_i[pad]])
        pts_sub = pts_sub.unsqueeze(0)    # (1, 8192, 3)

        valid = torch.ones(1, pts_sub.shape[1], dtype=torch.bool, device=device)
        pts_sub_d = pts_sub.to(device)
        pts_norm, _ = normalize_input(pts_sub_d, valid, pts_sub_d, valid, mode=norm_mode)

        with torch.no_grad():
            enc = nova_model._encode(pointmaps=pts_norm, test=True)
            z_i = enc["tokens"].float().cpu()   # (1, 768, 128)

        vis_pts_world_list.append(pts_world.numpy())
        vis_pts_cam_list.append(pts_sub.cpu())
        vis_pts_norm_list.append(pts_norm.cpu())
        z_star_list.append(z_i)

        print(f"  frame {fid:06d}: {n:,} vis pts  →  z_star {tuple(z_i.shape)}")

    # Stack to batched tensors
    da3_batch   = da3_agg.view(T_frames, N_per_frame, D_da3).to(device)  # (T, N_per_frame, D)
    z_star_batch= torch.cat(z_star_list, dim=0).to(device)               # (T, 768, 128)
    print(f"[perframe] da3_batch:    {tuple(da3_batch.shape)}")
    print(f"[perframe] z_star_batch: {tuple(z_star_batch.shape)}")

    # ── adapter ───────────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
    ).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(adapter.parameters(),
                                  lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # ── wandb ─────────────────────────────────────────────────────────────────
    run = None
    if use_wandb:
        import wandb
        run_name = f"overfit_perframe_stride{args.stride}_{time.strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(
            project = cfg.wandb_project,
            name    = run_name,
            config  = {
                "mode":            "per_frame",
                "stride":          args.stride,
                "steps":           int(cfg.steps),
                "lr":              float(cfg.lr),
                "weight_decay":    float(cfg.weight_decay),
                "hidden_dim":      int(cfg.hidden_dim),
                "depth":           int(cfg.depth),
                "num_heads":       int(cfg.num_heads),
                "source_dim":      int(cfg.source_dim),
                "target_tokens":   int(cfg.target_tokens),
                "target_dim":      int(cfg.target_dim),
                "n_params":        n_params,
                "n_per_frame_tokens": N_per_frame,
                "num_frames":      T_frames,
                "num_queries_decode": args.num_queries,
                "room":            room,
            },
            tags = ["overfit", "per-frame", "single-scene"],
        )
        print(f"[wandb] {run.url}")

    # ── log RGB frames ────────────────────────────────────────────────────────
    print("[perframe] Logging RGB frames …")
    tab10 = matplotlib.colormaps["tab10"]
    frame_imgs_wandb = []
    fig_grid, axes = plt.subplots(2, 4, figsize=(20, 7))
    for ax, fid, i in zip(axes.flat, frame_ids, range(T_frames)):
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        if use_wandb:
            frame_imgs_wandb.append(wandb.Image(img, caption=f"frame {fid:06d}"))
        ax.imshow(img)
        ax.set_title(f"frame {fid:06d}", fontsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor(tab10(i / max(T_frames-1,1))); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    fig_grid.suptitle(f"8 frames  (stride={args.stride})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"frames/grid": fig_to_wandb(fig_grid, use_wandb=use_wandb),
                   "frames/individual": frame_imgs_wandb}, step=0)
    else:
        plt.close(fig_grid)

    # ── log per-frame input pointclouds (in each frame's own camera frame) ───
    print("[perframe] Logging per-frame input pointclouds …")
    input_pc_wandb = []
    for i, fid in enumerate(frame_ids):
        pts_c = vis_pts_cam_list[i][0].numpy()   # (8192, 3) in frame i's camera
        cam_origin = np.zeros(3)                  # camera is at origin in its own frame
        fig = render_pts(pts_c, pts_c[:, 2], "plasma",
                         f"Frame {fid:06d} input (own-camera frame, colored by depth)",
                         cam_pos=cam_origin, cbar_label="Z depth (m)")
        if use_wandb:
            input_pc_wandb.append(fig_to_wandb(fig, f"input PC frame {fid:06d}", use_wandb=use_wandb))
        else:
            plt.close(fig)
    if use_wandb:
        wandb.log({"pointclouds/input_per_frame": input_pc_wandb}, step=0)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"[perframe] Training {int(cfg.steps)} steps on batch of T={T_frames} frames …")
    adapter.train()
    t0 = time.time()

    for step in range(1, int(cfg.steps) + 1):
        # Batch over all T frames simultaneously
        pred = adapter(da3_batch)                       # (T, 768, 128)
        loss = F.mse_loss(pred, z_star_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        if step % int(cfg.log_every) == 0 or step == 1 or step == int(cfg.steps):
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{cfg.steps}  loss={loss_val:.6f}  t={elapsed:.1f}s")
            if use_wandb:
                wandb.log({"train/loss": loss_val, "train/step": step}, step=step)

    print(f"[perframe] Training done. Final loss: {loss_val:.6f}")

    ckpt_path = ckpt_dir / "overfit_perframe_final.pt"
    torch.save({"adapter_state_dict": adapter.state_dict(),
                "config": OmegaConf.to_container(cfg),
                "final_loss": loss_val, "steps": int(cfg.steps),
                "n_per_frame_tokens": N_per_frame}, ckpt_path)
    if use_wandb:
        wandb.log({"train/final_loss": loss_val}, step=int(cfg.steps))

    # ── per-frame evaluation ──────────────────────────────────────────────────
    adapter.eval()
    with torch.no_grad():
        pred_batch = adapter(da3_batch)   # (T, 768, 128)

    mse_per_frame = F.mse_loss(pred_batch, z_star_batch, reduction="none")
    mse_per_frame = mse_per_frame.mean(dim=[1, 2])   # (T,)

    print(f"\n[perframe] Running per-frame decode + Chamfer …")

    cd_pred_vs_gt_list      = []
    cd_ae_vs_input_list     = []
    cd_pred_vs_input_list   = []
    gt_dec_wandb   = []
    pred_dec_wandb = []
    comp_figs      = []

    for i, fid in enumerate(frame_ids):
        z_i    = z_star_list[i].to(device)      # (1, 768, 128)
        pr_i   = pred_batch[i:i+1]              # (1, 768, 128)
        pn_i   = vis_pts_norm_list[i].to(device) # (1, 8192, 3) normalized

        print(f"  frame {fid:06d}: decoding …", end=" ", flush=True)
        gt_dec_i   = decode_tokens(nova_model, nova_cfg, z_i,  pn_i,
                                   device, args.num_queries, seed=42+i)  # (1, N, 3)
        pred_dec_i = decode_tokens(nova_model, nova_cfg, pr_i, pn_i,
                                   device, args.num_queries, seed=42+i)  # (1, N, 3)

        gt_np   = gt_dec_i[0].numpy()    # (N, 3)
        pr_np   = pred_dec_i[0].numpy()
        in_np   = vis_pts_norm_list[i][0].numpy()  # (8192, 3) normalized input

        cd_pg = chamfer_distance(torch.from_numpy(pr_np), torch.from_numpy(gt_np))
        cd_ai = chamfer_distance(torch.from_numpy(gt_np), torch.from_numpy(in_np))
        cd_pi = chamfer_distance(torch.from_numpy(pr_np), torch.from_numpy(in_np))

        cd_pred_vs_gt_list.append(cd_pg)
        cd_ae_vs_input_list.append(cd_ai)
        cd_pred_vs_input_list.append(cd_pi)

        print(f"CD(pred↔gt)={cd_pg:.4f}  CD(ae↔in)={cd_ai:.4f}  CD(pred↔in)={cd_pi:.4f}")

        if use_wandb:
            wandb.log({
                f"eval_perframe/frame{fid:06d}/mse":             float(mse_per_frame[i]),
                f"eval_perframe/frame{fid:06d}/cd_pred_vs_gt":   cd_pg,
                f"eval_perframe/frame{fid:06d}/cd_ae_vs_input":  cd_ai,
                f"eval_perframe/frame{fid:06d}/cd_pred_vs_input":cd_pi,
            }, step=int(cfg.steps))

        # Visualise GT decoded
        fig_gt = render_pts(gt_np, gt_np[:, 2], "viridis",
                            f"Frame {fid:06d} — GT decoded\n"
                            f"CD(ae↔in)={cd_ai:.4f}",
                            cbar_label="Z depth (normalized)")
        if use_wandb:
            gt_dec_wandb.append(fig_to_wandb(fig_gt, f"gt decoded frame {fid:06d}", use_wandb=use_wandb))
        else:
            plt.close(fig_gt)

        # Visualise pred decoded
        fig_pr = render_pts(pr_np, pr_np[:, 2], "plasma",
                            f"Frame {fid:06d} — Pred decoded\n"
                            f"CD(pred↔gt)={cd_pg:.4f}",
                            cbar_label="Z depth (normalized)")
        if use_wandb:
            pred_dec_wandb.append(fig_to_wandb(fig_pr, f"pred decoded frame {fid:06d}", use_wandb=use_wandb))
        else:
            plt.close(fig_pr)

        # Side-by-side for this frame
        fig_c, axs = plt.subplots(1, 3, figsize=(27, 7),
                                  subplot_kw={"projection": "3d"})
        for ax_c, pts_c, cmap_c, ttl_c in zip(
            axs,
            [in_np, gt_np, pr_np],
            ["cividis", "viridis", "plasma"],
            [f"Input pts (normalized)",
             f"GT decoded  CD_ae={cd_ai:.4f}",
             f"Pred decoded  CD={cd_pg:.4f}"],
        ):
            color_c = pts_c[:, 2]
            n_c = Normalize(np.percentile(color_c, 2), np.percentile(color_c, 98))
            col_c = matplotlib.colormaps[cmap_c](n_c(color_c))
            stp = max(1, len(pts_c) // 10_000)
            ax_c.scatter(pts_c[::stp, 0], pts_c[::stp, 2], pts_c[::stp, 1],
                         c=col_c[::stp], s=1.0, alpha=0.75)
            ax_c.set_title(ttl_c, fontsize=9, fontweight="bold")
            ax_c.set_xlabel("X"); ax_c.set_ylabel("Z"); ax_c.set_zlabel("Y")
            ax_c.view_init(elev=25, azim=-60)
            ax_c.tick_params(labelsize=7)
        fig_c.suptitle(
            f"Frame {fid:06d} | stride={args.stride} | "
            f"MSE={float(mse_per_frame[i]):.2e}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        if use_wandb:
            comp_figs.append(fig_to_wandb(fig_c, f"comparison frame {fid:06d}", use_wandb=use_wandb))
        else:
            plt.close(fig_c)

    if use_wandb:
        wandb.log({
            "pointclouds/gt_decoded_per_frame":   gt_dec_wandb,
            "pointclouds/pred_decoded_per_frame": pred_dec_wandb,
            "pointclouds/comparison_per_frame":   comp_figs,
        }, step=int(cfg.steps))

    # ── aggregate metrics ─────────────────────────────────────────────────────
    cd_pg_mean  = float(np.mean(cd_pred_vs_gt_list))
    cd_ai_mean  = float(np.mean(cd_ae_vs_input_list))
    cd_pi_mean  = float(np.mean(cd_pred_vs_input_list))
    mse_mean    = float(mse_per_frame.mean())

    if use_wandb:
        wandb.log({
            "eval_aggregate/cd_pred_vs_gt_mean":    cd_pg_mean,
            "eval_aggregate/cd_ae_vs_input_mean":   cd_ai_mean,
            "eval_aggregate/cd_pred_vs_input_mean": cd_pi_mean,
            "eval_aggregate/mse_mean":              mse_mean,
        }, step=int(cfg.steps))

    # Bar chart: per-frame Chamfer
    fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
    x = np.arange(T_frames)
    w = 0.25
    fid_labels = [f"{fid:06d}" for fid in frame_ids]
    ax_bar.bar(x - w, cd_pred_vs_gt_list,  w, label="CD pred↔gt decoded", color="#e76f51")
    ax_bar.bar(x,     cd_ae_vs_input_list, w, label="CD AE (gt↔input)",   color="#2a9d8f")
    ax_bar.bar(x + w, cd_pred_vs_input_list, w, label="CD pred↔input",    color="#457b9d")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(fid_labels, rotation=30, fontsize=8)
    ax_bar.set_ylabel("Chamfer distance")
    ax_bar.set_title(f"Per-frame Chamfer distances  (stride={args.stride})",
                     fontweight="bold")
    ax_bar.legend(fontsize=9); ax_bar.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if use_wandb:
        wandb.log({"eval_aggregate/chamfer_barchart": fig_to_wandb(fig_bar, "per-frame CD bar chart", use_wandb=use_wandb)},
                  step=int(cfg.steps))
    else:
        plt.close(fig_bar)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PER-FRAME  STRIDE {args.stride}  RESULTS")
    print(f"{'='*65}")
    header = f"  {'frame':>10}  {'MSE':>10}  {'CD pred↔gt':>12}  {'CD ae↔in':>10}  {'CD pred↔in':>12}"
    print(header)
    for i, fid in enumerate(frame_ids):
        print(f"  {fid:10d}  {float(mse_per_frame[i]):10.6f}  "
              f"{cd_pred_vs_gt_list[i]:12.6f}  "
              f"{cd_ae_vs_input_list[i]:10.6f}  "
              f"{cd_pred_vs_input_list[i]:12.6f}")
    print(f"  {'MEAN':>10}  {mse_mean:10.6f}  "
          f"{cd_pg_mean:12.6f}  {cd_ai_mean:10.6f}  {cd_pi_mean:12.6f}")
    print(f"{'='*65}")
    if use_wandb:
        print(f"  W&B: {run.url}")
        wandb.finish()
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
