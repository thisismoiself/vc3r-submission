#!/usr/bin/env python3
"""
Feature distribution analysis across Replica rooms.

For each room, loads the cached DA3 tokens and z_star (NOVA3R) tokens,
then answers: is room2 (test) inside or outside the training distribution?

Produces:
  analysis/pca_da3.png          – PCA of mean-pooled DA3 tokens per sample
  analysis/pca_zstar.png        – PCA of mean-pooled z_star tokens per sample
  analysis/tsne_da3.png         – t-SNE of DA3 features
  analysis/tsne_zstar.png       – t-SNE of z_star features
  analysis/stats_table.txt      – per-room statistics
  analysis/cosine_sim_matrix.png – pairwise cosine similarity between room means

Usage:
  python analyze_features.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

REPO_ROOT  = Path(__file__).resolve().parents[2]
EXP_DIR    = REPO_ROOT / "experiments" / "overfit_8frames"
DATA_ROOT  = EXP_DIR / "data" / "multi_scene_N50_s20"
OUT_DIR    = EXP_DIR / "analysis"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_ROOMS = ["room0", "room1", "office0", "office1", "office2", "office3", "office4"]
TEST_ROOMS  = ["room2"]
ALL_ROOMS   = TRAIN_ROOMS + TEST_ROOMS
N_SAMPLES   = 50

# Distinct palette – test room always red
COLORS = {
    "room0":   "#4e79a7",
    "room1":   "#f28e2b",
    "room2":   "#e15759",   # TEST
    "office0": "#76b7b2",
    "office1": "#59a14f",
    "office2": "#edc948",
    "office3": "#b07aa1",
    "office4": "#ff9da7",
}
MARKERS = {r: ("*" if r in TEST_ROOMS else "o") for r in ALL_ROOMS}


# ── load ──────────────────────────────────────────────────────────────────────

def load_all(rooms, n_samples):
    """Returns dicts of {room: np.ndarray}  shape (n_samples, feat_dim)."""
    da3_by_room   = {}
    zstar_by_room = {}

    for room in rooms:
        da3_list, zstar_list = [], []
        for si in range(n_samples):
            d = DATA_ROOT / room / f"sample_{si:03d}"
            da3   = torch.load(d / "da3_tokens.pt",   map_location="cpu", weights_only=True)
            zstar = torch.load(d / "z_star.pt",       map_location="cpu", weights_only=True)
            # da3:   (1, T*N_tok, 2048) → mean-pool → (2048,)
            # zstar: (1, 768, 128)      → mean-pool → (128,)
            da3_list.append(  da3[0].mean(0).numpy()   )   # (2048,)
            zstar_list.append(zstar[0].mean(0).numpy() )   # (128,)

        da3_by_room[room]   = np.stack(da3_list)     # (N, 2048)
        zstar_by_room[room] = np.stack(zstar_list)   # (N, 128)
        print(f"  {room}: loaded {n_samples} samples")

    return da3_by_room, zstar_by_room


# ── helpers ───────────────────────────────────────────────────────────────────

def stack_with_labels(by_room):
    X, labels, room_ids = [], [], []
    for ri, (room, feats) in enumerate(by_room.items()):
        X.append(feats)
        labels.extend([room] * len(feats))
        room_ids.extend([ri] * len(feats))
    return np.concatenate(X, axis=0), labels, np.array(room_ids)


def scatter_2d(ax, coords, labels, title, legend=True):
    for room in ALL_ROOMS:
        mask = np.array(labels) == room
        if not mask.any():
            continue
        is_test = room in TEST_ROOMS
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[room], marker=MARKERS[room],
                   s=(90 if is_test else 35),
                   alpha=0.85, linewidths=0.4, edgecolors="k" if is_test else "none",
                   label=f"{room} ({'TEST' if is_test else 'train'})", zorder=3 if is_test else 2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    if legend:
        ax.legend(fontsize=8, markerscale=1.4, framealpha=0.9)
    ax.grid(alpha=0.3)


# ── PCA ───────────────────────────────────────────────────────────────────────

def plot_pca(by_room, tag, title_prefix, out_path, explained_text=True):
    X, labels, _ = stack_with_labels(by_room)
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[1]))
    pca.fit(Xs)
    ev  = pca.explained_variance_ratio_

    # Project train-only, then project test onto the same basis
    train_mask = np.array([l not in TEST_ROOMS for l in labels])
    pca2 = PCA(n_components=2)
    pca2.fit(Xs[train_mask])
    coords = pca2.transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: 2D scatter
    scatter_2d(axes[0], coords, labels,
               f"{title_prefix} — PCA (train basis)")
    axes[0].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")

    # Right: cumulative variance + per-room reconstruction error
    ax2 = axes[1]
    ax2.plot(np.cumsum(ev[:20]) * 100, marker="o", ms=4, color="#264653")
    ax2.axhline(90, color="red", ls="--", lw=1, label="90 %")
    ax2.axhline(95, color="orange", ls="--", lw=1, label="95 %")
    ax2.set_xlabel("# PCA components")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_title("Variance explained (all rooms)", fontsize=11)
    ax2.legend(); ax2.grid(alpha=0.3)

    # Per-room distance from train centroid (in PCA space)
    pca50 = PCA(n_components=min(50, X.shape[1]))
    pca50.fit(Xs[train_mask])
    coords50 = pca50.transform(Xs)
    train_centroid = coords50[train_mask].mean(0)

    fig2, ax = plt.subplots(figsize=(10, 5))
    room_list = list(by_room.keys())
    means, stds = [], []
    for room in room_list:
        mask = np.array(labels) == room
        dists = np.linalg.norm(coords50[mask] - train_centroid[None], axis=1)
        means.append(dists.mean())
        stds.append(dists.std())
    colors_bar = [COLORS[r] for r in room_list]
    bars = ax.bar(room_list, means, yerr=stds, color=colors_bar,
                  capsize=4, edgecolor="k", linewidth=0.6)
    for bar, room in zip(bars, room_list):
        if room in TEST_ROOMS:
            bar.set_edgecolor("red"); bar.set_linewidth(2.5)
    ax.set_ylabel(f"L2 distance from train centroid\n(top-50 PCA space)")
    ax.set_title(f"{title_prefix} — Distance from training distribution", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(str(out_path).replace(".png", "_distance.png"), dpi=150)
    plt.close(fig2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Print per-room distance from centroid
    print(f"\n  [{tag}] L2 distance from train centroid (top-50 PCA):")
    for room, m, s in zip(room_list, means, stds):
        flag = " ← TEST" if room in TEST_ROOMS else ""
        print(f"    {room:>10}: {m:.4f} ± {s:.4f}{flag}")

    return ev


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def plot_tsne(by_room, tag, title_prefix, out_path, perplexity=30):
    X, labels, _ = stack_with_labels(by_room)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    # First reduce with PCA to 50 dims for speed
    pre = PCA(n_components=min(50, X.shape[1])).fit_transform(Xs)
    print(f"  [{tag}] Running t-SNE …")
    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                  max_iter=1000, verbose=0)
    coords = tsne.fit_transform(pre)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_2d(ax, coords, labels, f"{title_prefix} — t-SNE (perplexity={perplexity})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── cosine similarity matrix ──────────────────────────────────────────────────

def plot_cosine_matrix(by_room, tag, title_prefix, out_path):
    rooms = list(by_room.keys())
    means = np.stack([by_room[r].mean(0) for r in rooms])
    # Normalise
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    normed = means / (norms + 1e-8)
    sim = normed @ normed.T   # (R, R)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(rooms))); ax.set_yticks(range(len(rooms)))
    ax.set_xticklabels(rooms, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(rooms, fontsize=10)
    for i in range(len(rooms)):
        for j in range(len(rooms)):
            ax.text(j, i, f"{sim[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < sim[i,j] < 0.8 else "white")
    # Highlight test rows/cols
    for ri, room in enumerate(rooms):
        if room in TEST_ROOMS:
            for spine in ax.spines.values():
                spine.set_linewidth(0)
            ax.add_patch(plt.Rectangle((-0.5, ri-0.5), len(rooms), 1,
                                        fill=False, edgecolor="red", lw=2.5))
            ax.add_patch(plt.Rectangle((ri-0.5, -0.5), 1, len(rooms),
                                        fill=False, edgecolor="red", lw=2.5))
    plt.colorbar(im, ax=ax, fraction=0.046, label="Cosine similarity")
    ax.set_title(f"{title_prefix} — Pairwise cosine sim (room means)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

    print(f"\n  [{tag}] Cosine similarity of test rooms to each train room:")
    for test_room in TEST_ROOMS:
        ti = rooms.index(test_room)
        for room in TRAIN_ROOMS:
            ri = rooms.index(room)
            print(f"    {test_room} ↔ {room}: {sim[ti, ri]:.4f}")


# ── per-room statistics ───────────────────────────────────────────────────────

def print_stats(by_room, tag):
    lines = [f"\n{'='*60}", f"  {tag} — per-room statistics", f"{'='*60}"]
    lines.append(f"  {'room':>10}  {'mean_norm':>10}  {'std_norm':>10}  "
                  f"{'intra_cos':>10}  split")
    for room, feats in by_room.items():
        norms = np.linalg.norm(feats, axis=1)
        # Intra-room cosine similarity (mean pairwise among samples)
        normed = feats / (norms[:, None] + 1e-8)
        cos_mat = normed @ normed.T
        mask = ~np.eye(len(normed), dtype=bool)
        intra_cos = cos_mat[mask].mean()
        split = "TEST" if room in TEST_ROOMS else "train"
        lines.append(f"  {room:>10}  {norms.mean():10.4f}  {norms.std():10.4f}  "
                      f"{intra_cos:10.4f}  {split}")
    lines.append("="*60)
    text = "\n".join(lines)
    print(text)
    return text


# ── projection test: how well does the train PCA basis cover test room? ───────

def projection_coverage(da3_by_room):
    """Fraction of variance in test features captured by train PCA basis."""
    X, labels, _ = stack_with_labels(da3_by_room)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    train_mask = np.array([l not in TEST_ROOMS for l in labels])

    print("\n  [Coverage] Fraction of test variance captured by top-k train PCA components:")
    for k in [10, 20, 50, 100, 200]:
        pca = PCA(n_components=min(k, Xs[train_mask].shape[1]))
        pca.fit(Xs[train_mask])
        # Project test samples and compute reconstruction error
        X_test = Xs[~train_mask]
        X_rec  = pca.inverse_transform(pca.transform(X_test))
        total_var  = (X_test**2).sum()
        resid_var  = ((X_test - X_rec)**2).sum()
        coverage   = 1 - resid_var / total_var
        print(f"    k={k:4d}: {coverage*100:.1f}% coverage")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("[analyze] Loading cached features …")
    da3_by_room, zstar_by_room = load_all(ALL_ROOMS, N_SAMPLES)

    print("\n[analyze] DA3 token statistics …")
    stats_da3   = print_stats(da3_by_room,   "DA3 tokens  (mean-pooled, dim=2048)")
    print("\n[analyze] z_star statistics …")
    stats_zstar = print_stats(zstar_by_room, "z_star tokens (mean-pooled, dim=128)")

    projection_coverage(da3_by_room)

    print("\n[analyze] PCA plots …")
    plot_pca(da3_by_room,   "DA3",   "DA3 tokens",  OUT_DIR / "pca_da3.png")
    plot_pca(zstar_by_room, "zstar", "z_star (NOVA3R)", OUT_DIR / "pca_zstar.png")

    print("\n[analyze] Cosine similarity matrices …")
    plot_cosine_matrix(da3_by_room,   "DA3",   "DA3 tokens",     OUT_DIR / "cosine_da3.png")
    plot_cosine_matrix(zstar_by_room, "zstar", "z_star (NOVA3R)", OUT_DIR / "cosine_zstar.png")

    print("\n[analyze] t-SNE …")
    plot_tsne(da3_by_room,   "DA3",   "DA3 tokens",     OUT_DIR / "tsne_da3.png")
    plot_tsne(zstar_by_room, "zstar", "z_star (NOVA3R)", OUT_DIR / "tsne_zstar.png")

    # Save stats to file
    with open(OUT_DIR / "stats_table.txt", "w") as f:
        f.write(stats_da3 + "\n\n" + stats_zstar)
    print(f"\n[analyze] All outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
