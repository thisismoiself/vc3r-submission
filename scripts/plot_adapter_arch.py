"""DA3->NOVA3R adapter architecture, in the SAME style as plot_pointflow_arch.py:
a single clean pipeline row, with the DA3-token projection (source_proj) as its own Linear box."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/usr/prakt/s0016/vc3r/outputs/replica/adapter_arch"); OUT.mkdir(parents=True, exist_ok=True)

# identical palette + helpers to plot_pointflow_arch.py
FROZEN = "#cbd5e0"; FROZEN_E = "#718096"      # gray  = frozen
TRAIN  = "#c6f6d5"; TRAIN_E  = "#2f855a"      # green = trained
DATA   = "#ebf3fb"; DATA_E   = "#2b6cb0"      # blue  = tensors

def box(ax, x, y, w, h, text, fc, ec, fs=10, lw=1.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=3)

def arrow(ax, x0, y0, x1, y1, ec="#2d3748", lw=1.7, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                 lw=lw, color=ec, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))

fig, ax = plt.subplots(figsize=(12, 3.3))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")

y, h = 1.45, 0.95
box(ax, 0.15, y, 1.65, h, "DA3 tokens\n[B, 4096, 2048]",       DATA, DATA_E, 9)
box(ax, 2.15, y, 1.65, h, "source_proj\nLinear 2048$\\to$512", TRAIN, TRAIN_E, 9.5, lw=2.6)
box(ax, 4.15, y, 1.55, h, "projected source\n[B, 4096, 512]",  DATA, DATA_E, 9)
box(ax, 6.05, y, 2.0,  h, "4 $\\times$ CrossAttn\nBlock (512, 8h)", TRAIN, TRAIN_E, 9.5)
box(ax, 8.4,  y, 1.55, h, "LayerNorm +\nout_proj 512$\\to$128", TRAIN, TRAIN_E, 9.5)
box(ax, 10.3, y, 1.55, h, "NOVA3R tokens\n[B, 768, 128]",       DATA, DATA_E, 9)
for x0, x1 in [(1.80, 2.15), (3.80, 4.15), (5.70, 6.05), (8.05, 8.4), (9.95, 10.3)]:
    arrow(ax, x0, y + h/2, x1, y + h/2)

# learned queries feed the cross-attention from below
box(ax, 6.05, 0.2, 2.0, 0.66, "target_queries [768, 512]\nlearned", DATA, DATA_E, 8.4)
arrow(ax, 7.05, 0.86, 7.05, y)

# highlight tag on the linear box
ax.text(2.97, y - 0.28, "the DA3-token projection", ha="center", fontsize=8.5, color=TRAIN_E, weight="bold")

fig.suptitle(r"DA3$\to$NOVA3R adapter: DA3 tokens $\to$ Linear projection $\to$ cross-attention $\to$ 768$\times$128 tokens",
             y=1.02, fontsize=12.5)
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.86)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/adapter_arch.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {OUT}/adapter_arch.png / .pdf")
