#!/usr/bin/env python3
"""Re-render saved combined UMAP coordinates with presentation-sized points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dirs", type=Path, nargs="+")
    parser.add_argument("--point-size", type=float, default=30)
    args = parser.parse_args()

    for sample_dir in args.sample_dirs:
        coordinates = np.load(sample_dir / "umap_coordinates.npz")
        scene_embedding = coordinates["scene_ae"]
        adapter_embedding = coordinates["adapter"]
        metadata = json.loads((sample_dir / "metrics.json").read_text())
        title = f"{metadata['dataset']} - {metadata['split']}"

        both = np.concatenate([scene_embedding, adapter_embedding])
        lo, hi = both.min(0), both.max(0)
        padding = np.maximum((hi - lo) * 0.04, 1e-8)
        limits = (
            (lo[0] - padding[0], hi[0] + padding[0]),
            (lo[1] - padding[1], hi[1] + padding[1]),
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            scene_embedding[:, 0], scene_embedding[:, 1],
            s=args.point_size, alpha=0.55, edgecolors="none",
            color="#277da1", label="scene_ae",
        )
        ax.scatter(
            adapter_embedding[:, 0], adapter_embedding[:, 1],
            s=args.point_size, alpha=0.55, edgecolors="none",
            color="#f3722c", label="adapter",
        )
        ax.set(
            xlim=limits[0],
            ylim=limits[1],
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            title=f"{title} — scene_ae reference UMAP",
        )
        ax.legend(markerscale=1.5)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        output = sample_dir / "combined_umap_presentation.png"
        fig.savefig(output, dpi=240)
        plt.close(fig)
        print(f"[out] {output}")


if __name__ == "__main__":
    main()
