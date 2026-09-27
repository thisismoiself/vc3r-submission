#!/usr/bin/env python3
"""Fit UMAP on scene_ae tokens only, then transform adapter tokens."""

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

    scaler = StandardScaler().fit(scene_ae)
    scene_ae_scaled = scaler.transform(scene_ae)
    adapter_scaled = scaler.transform(adapter)

    pca = PCA(n_components=50, random_state=args.seed).fit(scene_ae_scaled)
    scene_ae_reduced = pca.transform(scene_ae_scaled)
    adapter_reduced = pca.transform(adapter_scaled)

    reducer = umap.UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
    ).fit(scene_ae_reduced)
    scene_ae_embedding = reducer.embedding_
    adapter_embedding = reducer.transform(adapter_reduced)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "coordinates.npz",
        scene_ae=scene_ae_embedding.astype(np.float32),
        adapter=adapter_embedding.astype(np.float32),
        window=np.array(args.window),
    )

    both = np.concatenate([scene_ae_embedding, adapter_embedding])
    x_min, y_min = both.min(axis=0)
    x_max, y_max = both.max(axis=0)
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_pad = 0.04 * max(y_max - y_min, 1e-8)
    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    sources = {
        "scene_ae": (scene_ae_embedding, "NOVA3R scene_ae (UMAP fit)", "#277da1"),
        "adapter": (adapter_embedding, "DA3 adapter (UMAP transform)", "#f3722c"),
    }
    for key, (embedding, title, color) in sources.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            s=14, alpha=0.55, edgecolors="none", color=color,
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_title(f"{title} — window {args.window}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{key}.png", dpi=240)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, (embedding, title, color) in sources.items():
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            s=12, alpha=0.45, edgecolors="none", color=color, label=title,
        )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_title(f"scene_ae-reference UMAP — window {args.window}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.out_dir / "combined.png", dpi=240)
    plt.close(fig)
    print(f"[out] {args.out_dir}")


if __name__ == "__main__":
    main()
