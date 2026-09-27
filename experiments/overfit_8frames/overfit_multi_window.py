#!/usr/bin/env python3
"""
Overfit the DA3→NOVA3R adapter on N cached windows, then decode the adapter's
prediction on a held-out validation window and show it in Rerun.

Default: train on data/multi_N10_s150/, validate on outputs/room0_stride150/.

Usage:
  python experiments/overfit_8frames/overfit_multi_window.py
  python experiments/overfit_8frames/overfit_multi_window.py --steps 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=Path,
                   default=Path(__file__).parent / "config.yaml")
    p.add_argument("--train-dir",   type=Path,
                   default=Path(__file__).parent / "data" / "multi_N10_s150",
                   help="Directory containing sample_000/ … with da3_tokens, z_star, pts_norm")
    p.add_argument("--val-dir",     type=Path,
                   default=REPO_ROOT / "outputs" / "room0_stride150",
                   help="Single-window cache to use as validation")
    p.add_argument("--steps",       type=int, default=2000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num-queries", type=int, default=8192)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--rrd-out",     type=Path,
                   default=REPO_ROOT / "outputs" / "overfit_stride150"
                           / "overfit10_val0.rrd")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor,
                  pts_norm: torch.Tensor,
                  device: torch.device,
                  num_queries: int, seed: int) -> np.ndarray:
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
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def norm_to_world(pts3d_norm: np.ndarray, norm_factor: float,
                  c2w: np.ndarray) -> np.ndarray:
    pts_cam = pts3d_norm / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


def main() -> None:
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    device = torch.device(args.device)

    # ── load training windows ──────────────────────────────────────────────────
    sample_dirs = sorted(args.train_dir.glob("sample_*"))
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_* dirs in {args.train_dir}")

    da3_list, zstar_list, train_starts = [], [], []
    for d in sample_dirs:
        m = torch.load(d / "meta.pt", map_location="cpu", weights_only=False)
        train_starts.append(int(m["start_idx"]))
        da3_list.append(  torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True))
        zstar_list.append(torch.load(d / "z_star.pt",     map_location="cpu", weights_only=True))

    da3_train   = torch.cat(da3_list,   dim=0).to(device)   # (N, T, D)
    zstar_train = torch.cat(zstar_list, dim=0).to(device)   # (N, 768, 128)
    N = da3_train.shape[0]
    print(f"[train] {N} windows  da3={tuple(da3_train.shape)}  z_star={tuple(zstar_train.shape)}")
    print(f"[train] start indices: {train_starts}")

    # ── load val window ────────────────────────────────────────────────────────
    val_meta   = torch.load(args.val_dir / "meta.pt",    map_location="cpu", weights_only=False)
    val_start  = int(val_meta.get("start_idx", val_meta["frame_ids"][0]))
    val_frames = set(int(f) for f in val_meta["frame_ids"])
    train_frames = set()
    for d in sample_dirs:
        m = torch.load(d / "meta.pt", map_location="cpu", weights_only=False)
        train_frames.update(int(f) for f in m["frame_ids"])
    overlap = val_frames & train_frames
    if overlap:
        raise ValueError(
            f"Val window (start={val_start}) shares {len(overlap)} frames with "
            f"training windows. Remove them before training.\nOverlapping: {sorted(overlap)[:10]}")
    print(f"[val]   start={val_start}  no frame overlap with training ✓")
    val_meta   = torch.load(args.val_dir / "meta.pt",     map_location="cpu", weights_only=False)
    val_zstar  = torch.load(args.val_dir / "z_star.pt",   map_location="cpu", weights_only=True)
    val_pnorm  = torch.load(args.val_dir / "pts_norm.pt", map_location="cpu", weights_only=True)
    val_da3    = torch.load(args.val_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)
    val_norm_factor = float(val_meta["norm_factor"])
    val_first_c2w   = val_meta["poses_c2w"][0].numpy()
    val_poses_c2w   = val_meta["poses_c2w"]
    print(f"[val]   start=0  frames {val_meta['frame_ids'][0]}-{val_meta['frame_ids'][-1]}  "
          f"norm_factor={val_norm_factor:.3f}m")

    # ── adapter ───────────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
        drop          = 0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"[train] {args.steps} steps  lr={args.lr}  N={N}")
    rng = torch.Generator().manual_seed(args.seed)
    adapter.train()
    for step in range(1, args.steps + 1):
        # shuffle indices each pass through
        idx = torch.randperm(N, generator=rng)
        pred = adapter(da3_train[idx])
        loss = F.mse_loss(pred, zstar_train[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # val MSE (no grad)
        if step % 200 == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                val_pred  = adapter(val_da3.to(device))
                val_loss  = F.mse_loss(val_pred, val_zstar.to(device)).item()
            print(f"  step {step:5d}/{args.steps}  train_loss={loss.item():.6f}  "
                  f"val_loss={val_loss:.6f}", flush=True)

    # final val prediction
    adapter.eval()
    with torch.no_grad():
        pred_tokens = adapter(val_da3.to(device)).cpu()   # (1, 768, 128)

    final_train = float(F.mse_loss(adapter(da3_train).cpu(), zstar_train.cpu()).item())
    final_val   = float(F.mse_loss(pred_tokens, val_zstar).item())
    print(f"\n[result] final train MSE: {final_train:.6f}  val MSE: {final_val:.6f}")

    # ── NOVA3R decode ──────────────────────────────────────────────────────────
    print("[decode] Loading NOVA3R …")
    ckpt_raw   = str(cfg.nova3r_ckpt)
    ckpt_local = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    ckpt_path  = ckpt_local if not Path(ckpt_raw).exists() else ckpt_raw
    nova_model, nova_cfg = load_nova3r_model(ckpt_path, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    pnorm_cpu = val_pnorm.cpu()

    print("[decode] GT z_star …")
    gt_norm   = decode_tokens(nova_model, nova_cfg, val_zstar,   pnorm_cpu,
                               device, args.num_queries, seed=args.seed)
    print("[decode] Adapter prediction …")
    pred_norm = decode_tokens(nova_model, nova_cfg, pred_tokens, pnorm_cpu,
                               device, args.num_queries, seed=args.seed)

    gt_world   = norm_to_world(gt_norm,   val_norm_factor, val_first_c2w)
    pred_world = norm_to_world(pred_norm, val_norm_factor, val_first_c2w)

    # un-normalised input conditioning points
    input_cam = pnorm_cpu[0].numpy() / 3.0 * val_norm_factor
    ones      = np.ones((len(input_cam), 1), dtype=np.float32)
    input_world = (val_first_c2w @ np.hstack([input_cam, ones]).T).T[:, :3]

    print(f"[decode] GT centroid:   {gt_world.mean(0).round(3)}")
    print(f"[decode] Pred centroid: {pred_world.mean(0).round(3)}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    import rerun as rr

    args.rrd_out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("overfit10_val0", spawn=False)
    rr.save(str(args.rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    BLUE   = np.array([ 50, 150, 255], dtype=np.uint8)
    GREEN  = np.array([ 60, 220,  90], dtype=np.uint8)
    GREY   = np.array([180, 180, 180], dtype=np.uint8)
    ORANGE = np.array([255, 140,  30], dtype=np.uint8)

    def sub(pts: np.ndarray, n: int = 150_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    rr.log("world/gt_zstar",
           rr.Points3D(sub(gt_world),
                       colors=np.tile(BLUE,   (min(len(gt_world),   150_000), 1)),
                       radii=0.012))
    rr.log("world/adapter_pred",
           rr.Points3D(sub(pred_world),
                       colors=np.tile(GREEN,  (min(len(pred_world), 150_000), 1)),
                       radii=0.012))
    rr.log("world/input_pts",
           rr.Points3D(sub(input_world, n=50_000),
                       colors=np.tile(GREY,   (min(len(input_world), 50_000), 1)),
                       radii=0.008))
    rr.log("world/cameras",
           rr.Points3D(val_poses_c2w[:, :3, 3].numpy(),
                       colors=np.tile(ORANGE, (len(val_poses_c2w), 1)),
                       radii=0.04))

    rr.log("legend", rr.TextDocument(
        "# 10-window overfit → val window (start=0)\n\n"
        f"- **Blue** `world/gt_zstar`: GT z_star decoded\n"
        f"- **Green** `world/adapter_pred`: adapter prediction decoded\n"
        "- **Grey** `world/input_pts`: input pts_norm conditioning\n"
        "- **Orange** `world/cameras`: 8 camera positions (stride 150)\n\n"
        f"Train: {N} windows  |  Val: start=0 frames {val_meta['frame_ids'][0]}–{val_meta['frame_ids'][-1]}\n\n"
        f"Steps: {args.steps}  |  Train MSE: {final_train:.2e}  |  Val MSE: {final_val:.2e}",
        media_type=rr.MediaType.MARKDOWN,
    ))

    print(f"\n[rerun] saved → {args.rrd_out}")
    print(f"  open with:  rerun {args.rrd_out}")


if __name__ == "__main__":
    main()
