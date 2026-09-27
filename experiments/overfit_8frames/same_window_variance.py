#!/usr/bin/env python3
"""Same-window variance of z_star.

Re-encode ONE fixed window many times under different sampling seeds.
The NOVA3R encoder (triposg_hybrid) builds its 768 query tokens from points
RANDOMLY SAMPLED from the cloud (_sample_features -> seeded rng.choice + FPS),
so the slot index = sampling order = arbitrary.

We show:
  1. Per-slot z_star changes a lot across seeds (identity distance high) ...
  2. ... but the SET is stable (Hungarian-matched distance much smaller) ...
  3. ... and the decoded pointcloud is essentially identical across seeds.
This is the within-window twin of the cross-window permutation result:
even for a SINGLE window there is no canonical slot ordering.
"""
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import scene_adapt as SA
from omegaconf import OmegaConf

DATA = EXP / "data" / "multi_scene_N200_s20" / "room0"
K_SEEDS = 8


@contextlib.contextmanager
def deterministic_numpy_default_rng(seed):
    original = np.random.default_rng
    np.random.default_rng = lambda s=None: original(seed if s is None else s)
    try:
        yield
    finally:
        np.random.default_rng = original


def pair_stats(za, zb, rng):
    T = za.shape[0]
    ident = np.linalg.norm(za - zb, axis=1).mean()
    rand = np.linalg.norm(za - zb[rng.permutation(T)], axis=1).mean()
    C = np.linalg.norm(za[:, None] - zb[None], axis=2)
    ri, ci = linear_sum_assignment(C)
    hung = C[ri, ci].mean()
    return ident, rand, hung


def chamfer(a, b):
    d = torch.cdist(a, b)
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(EXP / "config.yaml")
    print("Loading NOVA3R ...")
    nova_model, nova_cfg = SA.load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    pnorm = torch.load(DATA / "sample_000" / "pts_norm.pt",
                       map_location="cpu", weights_only=True).float().to(device)  # [1,8192,3]

    # re-encode same window under K different sampling seeds
    Zs = []
    for s in range(K_SEEDS):
        with deterministic_numpy_default_rng(s):
            z = nova_model._encode(pointmaps=pnorm, test=True)["tokens"].float()  # [1,768,128]
        Zs.append(z[0].cpu().numpy())
    Z = np.stack(Zs)  # [K,768,128]
    print(f"re-encoded same window {K_SEEDS}x -> {Z.shape}")

    rng = np.random.default_rng(0)
    ident, rand, hung = [], [], []
    for i in range(K_SEEDS):
        for j in range(i + 1, K_SEEDS):
            a, r, h = pair_stats(Z[i], Z[j], rng)
            ident.append(a); rand.append(r); hung.append(h)
    ident, rand, hung = map(np.array, (ident, rand, hung))

    print("\n=== same window, different sampling seed: token distances ===")
    print(f"  identity  (slot s vs slot s) : {ident.mean():.3f} +- {ident.std():.3f}")
    print(f"  random    permutation        : {rand.mean():.3f} +- {rand.std():.3f}")
    print(f"  Hungarian (optimal match)    : {hung.mean():.3f} +- {hung.std():.3f}")
    print(f"  identity/Hungarian ratio     : {ident.mean()/hung.mean():.2f}x")
    print(f"  |identity-random|/random     : {abs(ident.mean()-rand.mean())/rand.mean()*100:.1f}%")
    print("  => identity ~= random  >>  Hungarian : same SET, arbitrary ORDER, even for ONE window")

    # decoded pointclouds for two seeds: should match (set preserved)
    dec = lambda zz: SA.decode_tokens(
        nova_model, nova_cfg, torch.tensor(zz).unsqueeze(0), pnorm.cpu(), device,
        num_queries=4096, seed=42, ode_step_size=0.01)[0]
    p0, p1 = dec(Z[0]), dec(Z[1])
    print("\n=== decoded pointclouds across seeds ===")
    print(f"  chamfer(decode(seed0), decode(seed1)) = {chamfer(p0, p1):.4e} (normed units)")
    print("  (tiny => different per-slot tokens decode to the SAME geometry)")

    out = EXP / "ply_export" / "same_window_variance"
    out.mkdir(parents=True, exist_ok=True)
    SA.write_ply(out / "seed0.ply", p0.numpy())
    SA.write_ply(out / "seed1.ply", p1.numpy())
    print(f"\nPLYs -> {out}  (seed0.ply, seed1.ply)")


if __name__ == "__main__":
    main()
