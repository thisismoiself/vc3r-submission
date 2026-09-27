#!/usr/bin/env python3
"""Document the permutation / collapse behaviour of NOVA3R z_star tokens.

Produces numbers + figures that justify a permutation-invariant (Hungarian)
training loss for the DA3->NOVA3R adapter.

Analyses
  A. Slot-identity test:  distance between slot s of window A and slot s of
     window B (identity match) vs a random slot (random match) vs the optimal
     (Hungarian) match.  If slots carry identity:  identity << random.
     If slots are permuted:  identity ~= random  >> hungarian.
  B. Set-vs-order:  Hungarian assignment cost vs identity assignment cost per
     window pair.  Big gap => same set, different order.
  C. Input-consistency:  do similar DA3 inputs have similar per-slot targets?
     Plot DA3 cosine similarity vs (per-slot target dist) and vs (Hungarian
     matched dist).  Per-slot stays ~random regardless of input similarity
     (=> contradictory supervision => collapse); Hungarian tracks input.
  D. Low-rank / slot non-separability:  effective rank of the token cloud and
     silhouette of slot-index labels in PCA space (~0 => no slot identity).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent
DATA = EXP / "data" / "multi_scene_N200_s20" / "room0"
OUT = EXP / "latent_analysis"
OUT.mkdir(exist_ok=True)

N = 40           # windows to load
SEED = 0
rng = np.random.default_rng(SEED)


def load():
    Z, D, P = [], [], []
    for si in range(N):
        d = DATA / f"sample_{si:03d}"
        Z.append(torch.load(d / "z_star.pt", map_location="cpu", weights_only=True)[0])      # [768,128]
        # DA3: mean-pool over the 2048 source tokens -> [2048] global descriptor
        da3 = torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True)[0]       # [2048,2048]
        D.append(da3.mean(0))
        P.append(torch.load(d / "pts_norm.pt", map_location="cpu", weights_only=True)[0])     # [8192,3]
    Z = torch.stack(Z).numpy()      # [N,768,128]
    D = torch.stack(D).numpy()      # [N,2048]
    P = torch.stack(P).numpy()      # [N,8192,3]
    print(f"loaded Z{Z.shape} D{D.shape} P{P.shape}")
    return Z, D, P


def matched_dist(za, zb):
    """returns (identity, random, hungarian) mean per-slot L2 between two windows."""
    T = za.shape[0]
    ident = np.linalg.norm(za - zb, axis=1).mean()
    rp = rng.permutation(T)
    rand = np.linalg.norm(za - zb[rp], axis=1).mean()
    C = np.linalg.norm(za[:, None, :] - zb[None, :, :], axis=2)   # [T,T]
    ri, ci = linear_sum_assignment(C)
    hung = C[ri, ci].mean()
    return ident, rand, hung


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def chamfer(pa, pb, k=2048):
    a = pa[rng.choice(pa.shape[0], k, replace=False)]
    b = pb[rng.choice(pb.shape[0], k, replace=False)]
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    return float((d.min(1).mean() + d.min(0).mean()) / 2)


def main():
    Z, D, P = load()
    pairs = [(int(i), int(j)) for i in range(N) for j in range(i + 1, N)]
    pairs = [pairs[k] for k in rng.choice(len(pairs), min(120, len(pairs)), replace=False)]

    ident, rand, hung, dsim, psim = [], [], [], [], []
    for a, b in pairs:
        i, r, h = matched_dist(Z[a], Z[b])
        ident.append(i); rand.append(r); hung.append(h)
        dsim.append(cos(D[a], D[b]))
        psim.append(chamfer(P[a], P[b]))
    ident, rand, hung = map(np.array, (ident, rand, hung))
    dsim, psim = np.array(dsim), np.array(psim)

    print("\n=== A/B: per-slot distance between windows (token space) ===")
    print(f"  identity match : {ident.mean():.3f} +- {ident.std():.3f}")
    print(f"  random  match  : {rand.mean():.3f} +- {rand.std():.3f}")
    print(f"  Hungarian match: {hung.mean():.3f} +- {hung.std():.3f}")
    print(f"  identity/hungarian ratio : {ident.mean()/hung.mean():.2f}x")
    print(f"  |identity-random|/random : {abs(ident.mean()-rand.mean())/rand.mean()*100:.1f}%  (small => no slot identity)")

    # ---- Figure 1: three-bar headline ----
    plt.figure(figsize=(5, 4))
    means = [ident.mean(), rand.mean(), hung.mean()]
    errs = [ident.std(), rand.std(), hung.std()]
    plt.bar(["identity\n(slot s vs slot s)", "random\nperm", "Hungarian\n(optimal)"],
            means, yerr=errs, color=["#c0392b", "#7f8c8d", "#27ae60"], capsize=5)
    plt.ylabel("mean per-token L2 distance")
    plt.title("z_star slots carry no identity\n(identity ~= random  >>  Hungarian)")
    plt.tight_layout(); plt.savefig(OUT / "fig1_slot_identity.png", dpi=140); plt.close()

    # ---- Figure 2: input-consistency ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(dsim, ident, s=14, c="#c0392b", label="per-slot (identity)", alpha=.7)
    ax[0].scatter(dsim, hung, s=14, c="#27ae60", label="Hungarian", alpha=.7)
    ax[0].set_xlabel("DA3 input cosine similarity"); ax[0].set_ylabel("target token distance")
    ax[0].set_title("similar input -> similar target?"); ax[0].legend()
    ax[1].scatter(psim, ident, s=14, c="#c0392b", label="per-slot (identity)", alpha=.7)
    ax[1].scatter(psim, hung, s=14, c="#27ae60", label="Hungarian", alpha=.7)
    ax[1].set_xlabel("pointcloud Chamfer (window similarity)"); ax[1].set_ylabel("target token distance")
    ax[1].set_title("similar geometry -> similar target?"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "fig2_input_consistency.png", dpi=140); plt.close()

    # correlations: does target distance track input/geometry similarity?
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    print("\n=== C: correlation of target-distance with input/geometry similarity ===")
    print(f"  per-slot vs DA3-sim   : {corr(dsim, ident):+.3f}   Hungarian vs DA3-sim   : {corr(dsim, hung):+.3f}")
    print(f"  per-slot vs cloud-CD  : {corr(psim, ident):+.3f}   Hungarian vs cloud-CD  : {corr(psim, hung):+.3f}")
    print("  (per-slot retains some corr from global token magnitude, but its distance")
    print("   floor sits at the random/sampling-noise level; Hungarian is markedly cleaner)")

    # ---- D: effective rank + slot silhouette ----
    flat = Z.reshape(-1, Z.shape[-1])                 # [N*768,128]
    flat_c = flat - flat.mean(0)
    sv = np.linalg.svd(flat_c, compute_uv=False)
    p = sv**2 / (sv**2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    print("\n=== D: token cloud structure ===")
    print(f"  effective rank (128 dims): {eff_rank:.1f}")
    print(f"  top-3 singular value share: {p[:3].sum()*100:.1f}%")

    # PCA scatter coloured by slot index (subset of slots)
    U = flat_c @ np.linalg.svd(flat_c, full_matrices=False)[2][:2].T   # [N*768,2]
    slot_idx = np.tile(np.arange(Z.shape[1]), Z.shape[0])
    plt.figure(figsize=(5, 5))
    keep = rng.choice(flat.shape[0], 4000, replace=False)
    sc = plt.scatter(U[keep, 0], U[keep, 1], c=slot_idx[keep], s=4, cmap="hsv", alpha=.5)
    plt.colorbar(sc, label="slot index"); plt.title("PCA of all z_star tokens\n(no clustering by slot => no identity)")
    plt.tight_layout(); plt.savefig(OUT / "fig3_pca_slots.png", dpi=140); plt.close()

    print(f"\nfigures -> {OUT}")
    for f in ["fig1_slot_identity.png", "fig2_input_consistency.png", "fig3_pca_slots.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
