#!/usr/bin/env python3
"""
Generate documentation figures summarising the DA3→NOVA3R adapter experiments.

Outputs (all in analysis/):
  fig_results_summary.png    – bar chart: overfitting → multi-scene → test gap
  fig_feature_distribution.png – t-SNE + cosine matrix side-by-side
  fig_regularization_plan.png  – what each reg. technique targets
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = Path(__file__).parent / "analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_OVERFIT  = "#4e79a7"   # blue
C_TRAIN    = "#59a14f"   # green
C_TEST     = "#e15759"   # red
C_CEIL     = "#888888"   # grey (AE ceiling)

ROOM_COLORS = {
    "room0":   "#4e79a7", "room1":   "#f28e2b", "room2":   "#e15759",
    "office0": "#76b7b2", "office1": "#59a14f", "office2": "#edc948",
    "office3": "#b07aa1", "office4": "#ff9da7",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Results summary: single-room overfit vs multi-scene
# ═══════════════════════════════════════════════════════════════════════════════

def fig_results_summary():
    # Numbers from experiment logs
    experiments = {
        "Single-room\n(N=50, room0)\noverfitting":  dict(mse=6.63e-4, cd=2.865e-3, split="train"),
        "Multi-scene\ntrain rooms\n(room0, room1)": dict(mse=7.31e-3, cd=9.97e-4,  split="train"),
        "Multi-scene\ntest room\n(room2)":          dict(mse=2.265,   cd=4.16e-2,  split="test"),
        "NOVA3R AE\nceiling\n(test room)":          dict(mse=None,    cd=8.5e-3,   split="ceil"),
    }
    # AE ceiling CD is the avg cd_ae↔input for room2

    labels = list(experiments.keys())
    mse_vals = [v["mse"] for v in experiments.values()]
    cd_vals  = [v["cd"]  for v in experiments.values()]
    splits   = [v["split"] for v in experiments.values()]
    colors   = [C_OVERFIT if s == "train" else (C_TEST if s == "test" else C_CEIL)
                for s in splits]
    # override second bar
    colors[1] = C_TRAIN

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("DA3 → Q-Former Adapter: Experimental Results", fontsize=14, fontweight="bold")

    # ── MSE bar (skip AE ceiling, it has no MSE) ──────────────────────────────
    ax = axes[0]
    mse_labels = labels[:3]
    mse_data   = mse_vals[:3]
    mse_colors = colors[:3]
    bars = ax.bar(range(3), mse_data, color=mse_colors, edgecolor="k", linewidth=0.7, width=0.55)
    for bar, val in zip(bars, mse_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{val:.2e}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(mse_labels, fontsize=10)
    ax.set_ylabel("MSE (token space, log scale)", fontsize=11)
    ax.set_title("Latent space MSE", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.35, which="both")
    ax.annotate("300× gap", xy=(2, mse_data[2]), xytext=(1.6, mse_data[2] * 0.2),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                color="red", fontsize=9.5, fontweight="bold")

    # ── CD bar (all 4, AE ceiling as dashed line style) ──────────────────────
    ax2 = axes[1]
    cd_main   = cd_vals[:3]
    cd_colors = colors[:3]
    bars2 = ax2.bar(range(3), cd_main, color=cd_colors, edgecolor="k", linewidth=0.7, width=0.55)
    for bar, val in zip(bars2, cd_main):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                 f"{val*100:.2f} cm", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    # AE ceiling as horizontal line over test bar
    ax2.axhline(cd_vals[3], color=C_CEIL, ls="--", lw=2.0, zorder=5,
                label=f"NOVA3R AE ceiling  ({cd_vals[3]*100:.2f} cm)")
    ax2.set_yscale("log")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(mse_labels, fontsize=10)
    ax2.set_ylabel("Chamfer Distance (m, log scale)", fontsize=11)
    ax2.set_title("Chamfer Distance  pred ↔ GT decoded", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9.5, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.35, which="both")
    ax2.annotate("42× gap", xy=(2, cd_main[2]), xytext=(1.6, cd_main[2] * 0.3),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                 color="red", fontsize=9.5, fontweight="bold")

    legend_items = [
        mpatches.Patch(fc=C_OVERFIT, ec="k", label="Single-room overfit (train)"),
        mpatches.Patch(fc=C_TRAIN,   ec="k", label="Multi-scene (train rooms)"),
        mpatches.Patch(fc=C_TEST,    ec="k", label="Multi-scene (test room – unseen)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = OUT_DIR / "fig_results_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Feature distribution: t-SNE + cosine similarity
# ═══════════════════════════════════════════════════════════════════════════════

def fig_feature_distribution():
    """Re-uses pre-computed analysis PNGs and assembles a single figure."""
    analysis = Path(__file__).parent / "analysis"
    tsne_path   = analysis / "tsne_da3.png"
    cosine_path = analysis / "cosine_da3.png"
    dist_path   = analysis / "pca_da3_distance.png"

    if not tsne_path.exists():
        print(f"  Missing {tsne_path}, skipping fig_feature_distribution.")
        return

    from PIL import Image as PILImage

    fig = plt.figure(figsize=(17, 6.5))
    fig.suptitle("DA3 Feature Distribution Analysis  —  Why Generalisation Should Be Achievable",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(1, 3, figure=fig, wspace=0.05)

    def show(ax, path, title, note=""):
        img = np.asarray(PILImage.open(path).convert("RGB"))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        if note:
            ax.text(0.5, -0.03, note, ha="center", va="top",
                    transform=ax.transAxes, fontsize=9, color="#444", style="italic")

    show(fig.add_subplot(gs[0, 0]), tsne_path,
         "t-SNE of DA3 features (all rooms)",
         "room2 stars fully interleaved with train rooms")

    show(fig.add_subplot(gs[0, 1]), cosine_path,
         "Pairwise cosine similarity (room means)",
         "room2 ↔ train rooms: 0.994 – 0.998  (indistinguishable)")

    show(fig.add_subplot(gs[0, 2]), dist_path,
         "L2 distance from train centroid  (top-50 PCA)",
         "room2 distance (39.0) well within train room range")

    # Key finding annotation
    fig.text(0.5, 0.01,
             "Conclusion: room2 lies inside the training feature distribution. "
             "Failure to generalise is memorisation, not domain shift. "
             "Regularisation and more diverse windows should close the gap.",
             ha="center", fontsize=10.5, color="#c0392b", fontweight="bold",
             bbox=dict(fc="#fff5f5", ec="#c0392b", boxstyle="round,pad=0.4", lw=1.5))

    out = OUT_DIR / "fig_feature_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Regularisation strategy
# ═══════════════════════════════════════════════════════════════════════════════

def fig_regularization_plan():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    fig.suptitle("Regularisation Strategy for Multi-Scene Generalisation",
                 fontsize=13, fontweight="bold")

    rows = [
        ("Technique",             "What it does",
         "Why it helps",          "Config"),
        ("Random window starts",  "200 random 8-frame windows per room\n(not evenly-spaced)",
         "More intra-room trajectory diversity;\nprevent window-specific memorisation",
         "samples_per_room=200"),
        ("Q-Former dropout",      "drop=0.1 in attention weights\nand MLP layers",
         "Prevents individual attention heads\nfrom over-specialising on training scenes",
         "--drop 0.1"),
        ("Token subsampling",     "Random 4096/8288 patch tokens each step\n(different spatial view per step)",
         "Effectively 2× data augmentation;\nforces spatial invariance in cross-attention",
         "--tok-subsample 4096"),
        ("z★ label noise",        "Gaussian noise σ=0.01 on NOVA3R targets\nduring training",
         "Prevents exact memorisation of\ntraining z_star vectors",
         "--label-noise 0.01"),
    ]

    col_w = [0.18, 0.25, 0.32, 0.20]
    col_x = [0.02, 0.20, 0.46, 0.79]
    row_h = 0.16
    row_y_start = 0.88

    header_fc = "#264653"
    row_fcs   = ["#edf6f9", "#e8f5e9", "#fff3e0", "#fce4ec"]
    text_col  = "#111"

    for ri, row in enumerate(rows):
        y = row_y_start - ri * row_h
        for ci, (text, cw, cx) in enumerate(zip(row, col_w, col_x)):
            fc = header_fc if ri == 0 else row_fcs[(ri-1) % len(row_fcs)]
            tc = "white" if ri == 0 else text_col
            fs = 9.5 if ri == 0 else 9
            fw = "bold" if ri == 0 or ci == 0 else "normal"
            rect = mpatches.FancyBboxPatch((cx, y - row_h + 0.015), cw - 0.01, row_h - 0.02,
                                            boxstyle="round,pad=0.01",
                                            fc=fc, ec="#cccccc", lw=0.8,
                                            transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(cx + cw/2 - 0.005, y - row_h/2 + 0.015, text,
                    ha="center", va="center", fontsize=fs, color=tc,
                    fontweight=fw, transform=ax.transAxes,
                    multialignment="center")

    out = OUT_DIR / "fig_regularization_plan.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[make_figures] Generating documentation figures …")
    fig_results_summary()
    fig_feature_distribution()
    fig_regularization_plan()
    print(f"[make_figures] Done. All figures in {OUT_DIR}/")
