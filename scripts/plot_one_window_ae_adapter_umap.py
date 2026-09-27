#!/usr/bin/env python3
"""Joint scene_ae/adapter UMAP for one saved evaluation window."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.latents)
    scene_ae = data["scene_ae"][args.window]
    adapter = data["adapter"][args.window]
    if scene_ae.shape != adapter.shape:
        raise ValueError(f"Shape mismatch: {scene_ae.shape} vs {adapter.shape}")

    features = np.concatenate([scene_ae, adapter])
    labels = np.concatenate([
        np.zeros(len(scene_ae), dtype=np.int8),
        np.ones(len(adapter), dtype=np.int8),
    ])
    scaled = StandardScaler().fit_transform(features)
    reduced = PCA(n_components=50, random_state=args.seed).fit_transform(scaled)
    embedding = umap.UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        low_memory=True,
    ).fit_transform(reduced)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "coordinates.npz",
        embedding=embedding.astype(np.float32),
        labels=labels,
        window=np.array(args.window),
    )

    styles = {
        0: ("NOVA3R scene_ae", "#277da1"),
        1: ("DA3 adapter", "#f3722c"),
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for value, (name, color) in styles.items():
        mask = labels == value
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=12, alpha=0.48, edgecolors="none", color=color, label=name,
        )
    ax.set_title(f"One-window latent-token UMAP — window {args.window}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.out_dir / "overlay.png", dpi=240)
    plt.close(fig)

    x_min, y_min = embedding.min(axis=0)
    x_max, y_max = embedding.max(axis=0)
    x_pad = 0.04 * (x_max - x_min)
    y_pad = 0.04 * (y_max - y_min)
    for value, (name, color) in styles.items():
        mask = labels == value
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=12, alpha=0.48, edgecolors="none", color=color,
        )
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(f"{name} — window {args.window}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        key = "scene_ae" if value == 0 else "adapter"
        fig.savefig(args.out_dir / f"{key}.png", dpi=240)
        plt.close(fig)

    print(f"[out] {args.out_dir}")


if __name__ == "__main__":
    main()
