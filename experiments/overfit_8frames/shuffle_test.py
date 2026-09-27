#!/usr/bin/env python3
"""Permutation-invariance test for the NOVA3R decoder.

Take the z_star tokens of a real window, decode -> pts_a.
Shuffle the token order along dim=1, decode -> pts_b.
If the decoder is truly set-based over the tokens, pts_a == pts_b
(identical, since the ODE x_init uses a fixed seed).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import scene_adapt as SA  # reuse load_model wiring + decode_tokens


def chamfer_cm(a, b, scale=1.0):
    # a,b: [N,3] torch ; symmetric mean nearest-neighbour distance (not squared)
    d1 = torch.cdist(a, b).min(1).values.mean()
    d2 = torch.cdist(b, a).min(1).values.mean()
    return float((d1 + d2) / 2 * scale)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(EXP / "config.yaml")
    print("Loading NOVA3R ...")
    nova_model, nova_cfg = SA.load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    root = EXP / "data" / "multi_scene_N200_s20" / "room0" / "sample_000"
    z = torch.load(root / "z_star.pt", map_location="cpu", weights_only=True).float()  # [1,768,128]
    pnorm = torch.load(root / "pts_norm.pt", map_location="cpu", weights_only=True).float()  # [1,8192,3]
    print(f"z_star {tuple(z.shape)}  pnorm {tuple(pnorm.shape)}")

    # fixed permutation of the 768 token slots
    T = z.shape[1]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(T, generator=g)
    z_shuf = z[:, perm, :].clone()

    # sanity: it really is a permutation of the same set
    assert torch.allclose(z.sum(1), z_shuf.sum(1)), "not a permutation!"

    dec = lambda t: SA.decode_tokens(
        nova_model, nova_cfg, t, pnorm, device,
        num_queries=4096, seed=42, ode_step_size=0.01,
    )[0]  # [N,3]

    pts_a = dec(z)
    pts_b = dec(z_shuf)

    diff = (pts_a - pts_b).abs()
    print("\n=== identity-of-points (same x_init seed, point i vs point i) ===")
    print(f"  max |Δ|   = {diff.max().item():.3e}")
    print(f"  mean|Δ|   = {diff.mean().item():.3e}")
    print(f"  pts scale = [{pts_a.min().item():.2f}, {pts_a.max().item():.2f}]")
    print(f"  chamfer(pts_a, pts_b) = {chamfer_cm(pts_a, pts_b):.3e} (normed units)")

    # control: decode with a DIFFERENT (corrupted) set to show the metric is sensitive
    z_rand = torch.randn_like(z) * z.std() + z.mean()
    pts_c = dec(z_rand)
    print("\n=== control: decode random tokens (should differ a lot) ===")
    print(f"  chamfer(pts_a, pts_random) = {chamfer_cm(pts_a, pts_c):.3e} (normed units)")

    # second independent permutation, to show stability across different shuffles
    perm2 = torch.randperm(T, generator=torch.Generator().manual_seed(7))
    pts_b2 = dec(z[:, perm2, :].clone())

    out = EXP / "ply_export" / "shuffle_test"
    out.mkdir(parents=True, exist_ok=True)
    SA.write_ply(out / "orig.ply",        pts_a.numpy())
    SA.write_ply(out / "shuffled.ply",    pts_b.numpy())
    SA.write_ply(out / "shuffled2.ply",   pts_b2.numpy())
    SA.write_ply(out / "random_tokens.ply", pts_c.numpy())
    print(f"\nPLYs written to {out}")
    for f in ["orig.ply", "shuffled.ply", "shuffled2.ply", "random_tokens.ply"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
