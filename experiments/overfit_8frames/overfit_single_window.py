#!/usr/bin/env python3
"""
Overfit the DA3→NOVA3R adapter on a single wide-stride window, then decode
the GT z_star and the adapter prediction into a joint Rerun point cloud.

By default targets the stride-150 window cached at outputs/room0_stride150/.

Usage:
  ! python experiments/overfit_8frames/overfit_single_window.py
  ! python experiments/overfit_8frames/overfit_single_window.py --steps 500
  ! python experiments/overfit_8frames/overfit_single_window.py --window-dir outputs/room0_stride150 --steps 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
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
from depth_anything_3.cfg import create_object                             # noqa: E402
from safetensors.torch import load_file                                     # noqa: E402
from multi_scene_train import (                                             # noqa: E402
    load_da3_model,
    imagenet_normalize,
    normalize_extrinsics,
    select_tokens,
    extract_da3_tokens,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=Path,
                   default=Path(__file__).parent / "config.yaml")
    p.add_argument("--window-dir",  type=Path,
                   default=REPO_ROOT / "outputs" / "room0_stride150",
                   help="Directory containing z_star.pt, pts_norm.pt, meta.pt")
    p.add_argument("--replica-root", type=Path, default=None)
    p.add_argument("--steps",       type=int, default=1000)
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Higher LR is fine for single-sample overfit (default 1e-3)")
    p.add_argument("--num-queries", type=int, default=8192)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--rrd-out",     type=Path,
                   default=REPO_ROOT / "outputs" / "overfit_stride150"
                           / "overfit_vs_gt.rrd")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor,   # (1, 768, 128)
                  pts_norm: torch.Tensor, # (1, 8192, 3)
                  device: torch.device,
                  num_queries: int, seed: int) -> np.ndarray:
    """Decode tokens → (num_queries, 3) in normalised space."""
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
    device = torch.device(args.device)
    OmegaConf.set_struct(cfg, False)

    # ── load cached window ─────────────────────────────────────────────────────
    window_dir = args.window_dir
    meta = torch.load(window_dir / "meta.pt", map_location="cpu", weights_only=False)
    z_star    = torch.load(window_dir / "z_star.pt",   map_location="cpu", weights_only=True)  # (1,768,128)
    pts_norm  = torch.load(window_dir / "pts_norm.pt", map_location="cpu", weights_only=True)  # (1,8192,3)

    frame_ids   = meta["frame_ids"]
    poses_c2w   = meta["poses_c2w"].float()   # (8, 4, 4)
    first_c2w   = poses_c2w[0].numpy()
    norm_factor = float(meta["norm_factor"])
    room        = str(meta["room"])
    stride      = int(meta["stride"])

    print(f"[overfit] window: room={room}  stride={stride}  frames={frame_ids}")
    print(f"[overfit] norm_factor={norm_factor:.3f}m  z_star={tuple(z_star.shape)}")

    # ── extract DA3 tokens if not cached ──────────────────────────────────────
    da3_cache = window_dir / "da3_tokens.pt"
    if da3_cache.exists():
        da3_tokens = torch.load(da3_cache, map_location="cpu", weights_only=True)  # (1, N_tok, D)
        print(f"[overfit] DA3 tokens loaded from cache: {tuple(da3_tokens.shape)}")
    else:
        replica_root = (args.replica_root if args.replica_root is not None
                        else Path(str(cfg.replica_root)))
        room_dir     = replica_root / room
        results_dir  = room_dir / "results"
        with (replica_root / "cam_params.json").open() as f:
            cam_p = json.load(f)["camera"]
        H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
        K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                          [0., cam_p["fy"], cam_p["cy"]],
                          [0., 0., 1.]], dtype=np.float32)
        H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
        K_proc = K_nat.copy(); K_proc[0] *= W_proc / W_nat; K_proc[1] *= H_proc / H_nat

        def load_rgb(fid: int) -> torch.Tensor:
            img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
            img = img.resize((W_proc, H_proc), Image.BILINEAR)
            return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

        images     = torch.stack([load_rgb(fid) for fid in frame_ids])
        intrinsics = torch.from_numpy(
            np.tile(K_proc[None], (len(frame_ids), 1, 1))).float()

        print("[overfit] Loading DA3 model …")
        da3_model = load_da3_model(str(cfg.da3_model), device)

        da3_tokens = extract_da3_tokens(
            da3_model, images, poses_c2w, intrinsics,
            layer_idx=int(cfg.da3_source_layer_index),
            max_tokens=int(cfg.da3_max_source_tokens),
            device=device,
        )  # (1, N_tok, D)

        torch.save(da3_tokens, da3_cache)
        print(f"[overfit] DA3 tokens extracted + cached: {tuple(da3_tokens.shape)}")
        del da3_model
        torch.cuda.empty_cache()

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

    da3_dev  = da3_tokens.to(device)   # (1, N_tok, D)
    zstar_dev = z_star.to(device)      # (1, 768, 128)

    # ── overfit loop ──────────────────────────────────────────────────────────
    print(f"[overfit] Overfitting {args.steps} steps on 1 sample (lr={args.lr}) …")
    adapter.train()
    for step in range(1, args.steps + 1):
        pred = adapter(da3_dev)
        loss = F.mse_loss(pred, zstar_dev)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1 or step == args.steps:
            print(f"  step {step:5d}/{args.steps}  loss={loss.item():.6f}")

    adapter.eval()
    with torch.no_grad():
        pred_tokens = adapter(da3_dev).cpu()  # (1, 768, 128)

    final_loss = float(F.mse_loss(pred_tokens.to(device), zstar_dev).item())
    print(f"[overfit] Final MSE: {final_loss:.6f}")

    # ── decode both ───────────────────────────────────────────────────────────
    print("[overfit] Loading NOVA3R for decoding …")
    ckpt_raw   = str(cfg.nova3r_ckpt)
    ckpt_local = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    ckpt_path  = ckpt_local if not Path(ckpt_raw).exists() else ckpt_raw
    nova_model, nova_cfg = load_nova3r_model(ckpt_path, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    pts_norm_cpu = pts_norm.cpu()

    print("[overfit] Decoding GT z_star …")
    gt_norm  = decode_tokens(nova_model, nova_cfg, z_star,      pts_norm_cpu,
                              device, args.num_queries, seed=args.seed)
    print("[overfit] Decoding adapter prediction …")
    pred_norm = decode_tokens(nova_model, nova_cfg, pred_tokens, pts_norm_cpu,
                               device, args.num_queries, seed=args.seed)

    gt_world   = norm_to_world(gt_norm,   norm_factor, first_c2w)
    pred_world = norm_to_world(pred_norm, norm_factor, first_c2w)

    # un-normalise input pts_norm to world for reference
    input_pts_cam = pts_norm_cpu[0].numpy() / 3.0 * norm_factor
    ones = np.ones((len(input_pts_cam), 1), dtype=np.float32)
    input_world = (first_c2w @ np.hstack([input_pts_cam, ones]).T).T[:, :3]

    print(f"[overfit] GT world centroid:   {gt_world.mean(0)}")
    print(f"[overfit] Pred world centroid: {pred_world.mean(0)}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    import rerun as rr

    args.rrd_out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("overfit_single_window", spawn=False)
    rr.save(str(args.rrd_out))

    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    BLUE   = np.array([ 50, 150, 255], dtype=np.uint8)
    GREEN  = np.array([ 60, 220,  90], dtype=np.uint8)
    GREY   = np.array([180, 180, 180], dtype=np.uint8)
    ORANGE = np.array([255, 140,  30], dtype=np.uint8)

    def subsample(pts: np.ndarray, n: int = 150_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    rr.log(
        "world/gt_zstar",
        rr.Points3D(
            subsample(gt_world),
            colors=np.tile(BLUE, (min(len(gt_world), 150_000), 1)),
            radii=0.012,
        ),
    )
    rr.log(
        "world/adapter_pred",
        rr.Points3D(
            subsample(pred_world),
            colors=np.tile(GREEN, (min(len(pred_world), 150_000), 1)),
            radii=0.012,
        ),
    )
    rr.log(
        "world/input_pts",
        rr.Points3D(
            subsample(input_world, n=50_000),
            colors=np.tile(GREY, (min(len(input_world), 50_000), 1)),
            radii=0.008,
        ),
    )

    # Camera positions
    cam_positions = poses_c2w[:, :3, 3].numpy()
    rr.log(
        "world/cameras",
        rr.Points3D(cam_positions, colors=np.tile(ORANGE, (len(cam_positions), 1)),
                    radii=0.04),
    )

    rr.log(
        "legend",
        rr.TextDocument(
            "# Overfit single window\n\n"
            f"- **Blue** `world/gt_zstar`: GT NOVA3R z_star decoded ({args.num_queries:,} pts)\n"
            f"- **Green** `world/adapter_pred`: adapter prediction decoded ({args.num_queries:,} pts)\n"
            "- **Grey** `world/input_pts`: input pts_norm (mesh conditioning)\n"
            "- **Orange** `world/cameras`: 8 camera positions (stride 150)\n\n"
            f"Window: room={room}  stride={stride}  frames {frame_ids[0]}–{frame_ids[-1]}\n\n"
            f"Overfitting: {args.steps} steps  final MSE={final_loss:.2e}",
            media_type=rr.MediaType.MARKDOWN,
        ),
    )

    print(f"\n[overfit] saved → {args.rrd_out}")
    print(f"  open with:  rerun {args.rrd_out}")


if __name__ == "__main__":
    main()
