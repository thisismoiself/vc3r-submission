#!/usr/bin/env python3
"""
Step 2 — Overfit DA3→NOVA3R adapter on the 8-frame extracted data.

Goal: prove that a mapping from DA3 backbone tokens to NOVA3R AE latent
      tokens exists by driving MSE loss to near-zero on a single scene.

Logs every step to Weights & Biases and saves a final checkpoint.

Usage:
  python overfit.py [--config config.yaml] [--steps 3000] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DA3_SRC   = REPO_ROOT / "da3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402


# ── wandb helpers ─────────────────────────────────────────────────────────────

def init_wandb(cfg, run_name: str):
    try:
        import wandb
        run = wandb.init(
            project = cfg.wandb_project,
            name    = run_name,
            config  = {
                "steps":              cfg.steps,
                "lr":                 cfg.lr,
                "weight_decay":       cfg.weight_decay,
                "hidden_dim":         cfg.hidden_dim,
                "depth":              cfg.depth,
                "num_heads":          cfg.num_heads,
                "source_dim":         cfg.source_dim,
                "target_tokens":      cfg.target_tokens,
                "target_dim":         cfg.target_dim,
                "da3_source_layer":   cfg.da3_source_layer_index,
                "da3_max_tokens":     cfg.da3_max_source_tokens,
                "frame_stride":       cfg.frame_stride,
                "num_frames":         cfg.num_frames,
                "room":               cfg.room,
                "mesh_sample_points": cfg.mesh_sample_points,
            },
            tags = ["overfit", "single-scene", "proof-of-concept"],
        )
        print(f"[wandb] {run.url}")
        return run
    except Exception as e:
        print(f"[wandb] init failed: {e}")
        return None


def wandb_log(run, metrics: dict, step: int):
    if run is None:
        return
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        pass


def wandb_finish(run):
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overfit DA3→NOVA3R adapter on extracted 8-frame data"
    )
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override steps from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override data_dir (must contain da3_tokens.pt + z_star.pt)")
    parser.add_argument("--ckpt-dir", type=Path, default=None,
                        help="Override ckpt_dir for saving checkpoint")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override wandb run name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr

    device   = torch.device(args.device)
    data_dir = args.data_dir if args.data_dir is not None else Path(cfg.data_dir)
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    da3_tokens = torch.load(data_dir / "da3_tokens.pt", map_location=device,
                            weights_only=True)   # (1, T*N, D_da3)
    z_star     = torch.load(data_dir / "z_star.pt",    map_location=device,
                            weights_only=True)   # (1, 768, 128)

    print(f"[overfit] da3_tokens : {tuple(da3_tokens.shape)}")
    print(f"[overfit] z_star     : {tuple(z_star.shape)}")
    print(f"[overfit] device     : {device}")

    # ── adapter model ─────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim   = int(cfg.source_dim),
        hidden_dim   = int(cfg.hidden_dim),
        target_tokens= int(cfg.target_tokens),
        target_dim   = int(cfg.target_dim),
        depth        = int(cfg.depth),
        num_heads    = int(cfg.num_heads),
    ).to(device)

    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[overfit] Adapter params: {n_params:,}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr           = float(cfg.lr),
        weight_decay = float(cfg.weight_decay),
    )

    # ── wandb ─────────────────────────────────────────────────────────────────
    if args.run_name is not None:
        run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    elif hasattr(cfg, "wandb_run_name"):
        run_name = f"{cfg.wandb_run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = f"overfit_8frames_{time.strftime('%Y%m%d_%H%M%S')}"
    run = init_wandb(cfg, run_name)

    # ── training loop ─────────────────────────────────────────────────────────
    adapter.train()
    t0 = time.time()

    for step in range(1, int(cfg.steps) + 1):
        pred = adapter(da3_tokens)                             # (1, 768, 128)
        loss = F.mse_loss(pred, z_star)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        if step % int(cfg.log_every) == 0 or step == 1 or step == cfg.steps:
            elapsed = time.time() - t0
            print(f"step {step:5d}/{cfg.steps}  loss={loss_val:.6f}  t={elapsed:.1f}s")
            wandb_log(run, {"loss": loss_val, "step": step}, step=step)

    # ── save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = ckpt_dir / "overfit_final.pt"
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "config": OmegaConf.to_container(cfg),
        "final_loss": loss_val,
        "steps": int(cfg.steps),
        "da3_tokens_shape": list(da3_tokens.shape),
        "z_star_shape":     list(z_star.shape),
    }, ckpt_path)
    print(f"\n[overfit] Done. Final loss: {loss_val:.6f}")
    print(f"[overfit] Checkpoint: {ckpt_path}")

    wandb_log(run, {"final_loss": loss_val}, step=int(cfg.steps))
    wandb_finish(run)


if __name__ == "__main__":
    main()
