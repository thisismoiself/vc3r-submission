#!/usr/bin/env python3
"""Render separate, axis-matched panels from an existing joint UMAP embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.coordinates)
    embedding = data["embedding"]
    labels = data["labels"].astype(str)

    x_min, y_min = embedding.min(axis=0)
    x_max, y_max = embedding.max(axis=0)
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_pad = 0.04 * max(y_max - y_min, 1e-8)
    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    styles = {
        "scene_ae": ("NOVA3R scene_ae", "#277da1"),
        "scene_n1": ("NOVA3R scene_n1", "#43aa8b"),
        "adapter": ("DA3 adapter", "#f3722c"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, (title, color) in styles.items():
        mask = labels == key
        if not mask.any():
            raise ValueError(f"No coordinates found for {key}")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            alpha=0.24,
            edgecolors="none",
            color=color,
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_title(f"{title} latent-token UMAP")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        output = args.out_dir / f"{key}_umap.png"
        fig.savefig(output, dpi=240)
        plt.close(fig)
        print(f"[out] {key}: {int(mask.sum()):,} points -> {output}")


if __name__ == "__main__":
    main()
