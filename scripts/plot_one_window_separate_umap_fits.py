#!/usr/bin/env python3
"""Fit independent UMAP models to scene_ae and adapter tokens from one window."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


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
    arrays = {
        "scene_ae": data["scene_ae"][args.window],
        "adapter": data["adapter"][args.window],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, tokens in arrays.items():
        scaled = StandardScaler().fit_transform(tokens)
        reduced = PCA(n_components=50, random_state=args.seed).fit_transform(scaled)
        embedding = umap.UMAP(
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.seed,
            low_memory=True,
        ).fit_transform(reduced)

        color = "#277da1" if key == "scene_ae" else "#f3722c"
        title = "NOVA3R scene_ae" if key == "scene_ae" else "DA3 adapter"
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            s=14, alpha=0.55, edgecolors="none", color=color,
        )
        ax.set_title(f"{title} — independent UMAP, window {args.window}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{key}_independent_umap.png", dpi=240)
        plt.close(fig)
        np.savez_compressed(
            args.out_dir / f"{key}_coordinates.npz",
            embedding=embedding.astype(np.float32),
            window=np.array(args.window),
        )
        print(f"{key}: shape={tokens.shape} sha256={digest(tokens)}")

    scene_ae, adapter = arrays["scene_ae"], arrays["adapter"]
    difference = adapter - scene_ae
    print(f"exact_equal={np.array_equal(scene_ae, adapter)}")
    print(f"equal_elements={int(np.equal(scene_ae, adapter).sum())}/{scene_ae.size}")
    print(f"mse={float(np.mean(difference ** 2)):.9f}")
    print(f"max_abs_difference={float(np.max(np.abs(difference))):.9f}")


if __name__ == "__main__":
    main()
