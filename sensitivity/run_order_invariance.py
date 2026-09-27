#!/usr/bin/env python3
"""
Empirically test whether the NOVA3R decoder is permutation-invariant
with respect to z_star token ordering.

Decodes the same z_star with K random permutations of its 768 token slots
and measures Chamfer distance to the baseline decode. Also establishes a
decode stochasticity floor by decoding the unshuffled z_star twice with
different ODE solver seeds.

Outputs saved to sensitivity/val<N>/order_invariance.pt and a summary table.

Usage:
  ! python sensitivity/run_order_invariance.py
  ! python sensitivity/run_order_invariance.py --val-start 56 --n-shuffles 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT   = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model      # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper    # noqa: E402
from nova3r.flow_matching.solver import ODESolver            # noqa: E402
from nova3r.inference import amp_dtype_mapping               # noqa: E402

DATA_ROOT = REPO_ROOT / "scripts" / "data" / "windows"
OUT_ROOT  = REPO_ROOT / "sensitivity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-start",   type=int, default=56)
    p.add_argument("--n-shuffles",  type=int, default=10)
    p.add_argument("--num-queries", type=int, default=8192)
    p.add_argument("--decode-seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_window(start: int):
    win_dir   = DATA_ROOT / f"start_{start:04d}"
    consensus = torch.load(win_dir / "z_star_consensus.pt", map_location="cpu", weights_only=True)
    pts_norm  = torch.load(win_dir / "pts_norm.pt",         map_location="cpu", weights_only=True)
    return consensus, pts_norm


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor, pts_norm: torch.Tensor,
                  device: torch.device, num_queries: int, seed: int) -> np.ndarray:
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


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    dists  = torch.cdist(a_t.unsqueeze(0), b_t.unsqueeze(0))[0]
    a_to_b = dists.min(dim=1).values.mean()
    b_to_a = dists.min(dim=0).values.mean()
    return float((a_to_b + b_to_a) / 2)


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    print(f"[load] val window start={args.val_start}")
    z_consensus, pts_norm = load_window(args.val_start)

    print("[model] Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── Step 1: baseline decode ───────────────────────────────────────────────
    print("[step1] Baseline decode (seed=decode_seed) …")
    pts_ref = decode_tokens(nova_model, nova_cfg, z_consensus, pts_norm,
                            device, args.num_queries, seed=args.decode_seed)

    # ── Step 2: stochasticity floor ───────────────────────────────────────────
    print("[step2] Stochasticity floor (same z_star, different seed) …")
    pts_ref2   = decode_tokens(nova_model, nova_cfg, z_consensus, pts_norm,
                               device, args.num_queries, seed=args.decode_seed + 1)
    floor_dist = chamfer_distance(pts_ref, pts_ref2)
    print(f"  decode stochasticity floor: {floor_dist:.6f}")

    # ── Step 3: shuffled decodes ──────────────────────────────────────────────
    print(f"[step3] {args.n_shuffles} shuffled decodes …")
    shuffle_dists = []
    for k in range(args.n_shuffles):
        rng = torch.Generator().manual_seed(k)
        perm = torch.randperm(768, generator=rng)
        z_shuffled = z_consensus[:, perm, :]
        pts_shuffled = decode_tokens(nova_model, nova_cfg, z_shuffled, pts_norm,
                                     device, args.num_queries, seed=args.decode_seed)
        d = chamfer_distance(pts_shuffled, pts_ref)
        shuffle_dists.append(d)
        print(f"  shuffle {k:2d}  perm_seed={k}  Chamfer={d:.6f}", flush=True)

    # ── Step 4: report ────────────────────────────────────────────────────────
    shuffle_arr = np.array(shuffle_dists)
    print(f"\n[results]")
    print(f"  decode stochasticity floor : {floor_dist:.6f}")
    print(f"  shuffle Chamfer  mean      : {shuffle_arr.mean():.6f}")
    print(f"  shuffle Chamfer  std       : {shuffle_arr.std():.6f}")
    print(f"  shuffle Chamfer  min       : {shuffle_arr.min():.6f}")
    print(f"  shuffle Chamfer  max       : {shuffle_arr.max():.6f}")
    ratio = shuffle_arr.mean() / (floor_dist + 1e-9)
    print(f"  shuffle / floor ratio      : {ratio:.2f}x")

    if ratio < 2.0:
        verdict = "ORDER-AGNOSTIC — shuffle distances are at or near the decode noise floor."
    else:
        verdict = "ORDER-SENSITIVE — shuffle distances significantly exceed the noise floor."
    print(f"\n[verdict] {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = OUT_ROOT / f"val{args.val_start}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "floor_dist":   floor_dist,
        "shuffle_dists": shuffle_dists,
        "ratio":        float(ratio),
        "verdict":      verdict,
    }, out_dir / "order_invariance.pt")
    print(f"\n[done] → {out_dir}/order_invariance.pt")


if __name__ == "__main__":
    main()
