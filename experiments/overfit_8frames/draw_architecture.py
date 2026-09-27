#!/usr/bin/env python3
"""Render the DA3 → Q-Former Adapter → NOVA3R architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 17))
ax.set_xlim(0, 14)
ax.set_ylim(0, 17)
ax.axis("off")
fig.patch.set_facecolor("#ffffff")

# ── palette ───────────────────────────────────────────────────────────────────
FROZEN_FC  = "#dce8f5"
FROZEN_EC  = "#2e86c1"
ADAPT_FC   = "#fef9ef"
ADAPT_EC   = "#ca8a04"
BLOCK_FC   = "#fef3cd"
BLOCK_EC   = "#d4a017"
QUERY_FC   = "#fff0e0"
QUERY_EC   = "#d35400"
TENSOR_FC  = "#f5f5f5"
TENSOR_EC  = "#aaaaaa"
LOSS_FC    = "#fdecea"
LOSS_EC    = "#c0392b"
INPUT_FC   = "#f0f0f0"
INPUT_EC   = "#aaaaaa"
ARROW_C    = "#222222"
LOSS_C     = "#c0392b"

def rbox(cx, cy, w, h, fc, ec, lw=1.5, r=0.12, z=3):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle=f"round,pad={r}",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(p)

def txt(x, y, s, fs=10, bold=False, color="#111", z=6, ha="center", va="center", italic=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal")

def tensor(x, y, s):
    ax.text(x, y, s, ha="center", va="center", fontsize=8.2, color="#444",
            zorder=6, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.22", fc=TENSOR_FC, ec=TENSOR_EC, lw=0.9))

def arr(x1, y1, x2, y2, color=ARROW_C, lw=1.7, z=7, style="->"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"))

def darr(x1, y1, x2, y2, color=LOSS_C, lw=1.6, z=7, rad=0.0):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=(0,(5,3)),
                                connectionstyle=f"arc3,rad={rad}"))

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
txt(7, 16.6, "DA3  →  Q-Former Adapter  →  NOVA3R", fs=16, bold=True)
txt(7, 16.22, "Bridging image features to a 3-D scene autoencoder latent space",
    fs=10, italic=True, color="#555")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PATH  (DA3)    centre x = 4
# ═══════════════════════════════════════════════════════════════════════════════
LX = 4.0

# input
rbox(LX, 15.4, 5.2, 0.75, INPUT_FC, INPUT_EC, lw=1.1, r=0.10)
txt(LX, 15.57, "8 RGB Frames", fs=11, bold=True)
txt(LX, 15.22, "with camera poses  (c2w, intrinsics)", fs=9, color="#555", italic=True)

# DA3
arr(LX, 15.02, LX, 14.42)
rbox(LX, 14.05, 5.4, 0.75, FROZEN_FC, FROZEN_EC, lw=2.0)
txt(LX, 14.23, "DA3 Backbone", fs=12, bold=True, color="#1a4f7a")
txt(LX, 13.87, "multi-view ViT  ·  camera-conditioned  ·  frozen", fs=9, italic=True, color="#2e86c1")

arr(LX, 13.67, LX, 13.22)
tensor(LX, 13.07, "(B, 8, 1036, 2048)   —   8 frames × 1036 patch tokens × dim")

# subsample
arr(LX, 12.92, LX, 12.55)
rbox(LX, 12.33, 4.0, 0.40, "#eeeeee", "#999", lw=1.0, r=0.08)
txt(LX, 12.33, "flatten  ·  uniform subsample  →  2048 tokens", fs=9, color="#444")

arr(LX, 12.13, LX, 11.75)
tensor(LX, 11.60, "(B, 2048, 2048)   —   source tokens")

# ═══════════════════════════════════════════════════════════════════════════════
# Q-FORMER ADAPTER BOX
# ═══════════════════════════════════════════════════════════════════════════════
AY  = 9.55    # centre y of adapter box
AH  = 3.75    # height
AW  = 12.0    # width  — wide so queries and tokens don't overlap
ACX = 7.0     # centre x

arr(LX, 11.45, LX, AY + AH/2)

rbox(ACX, AY, AW, AH, ADAPT_FC, ADAPT_EC, lw=2.4, r=0.20, z=2)
txt(ACX, AY + AH/2 - 0.30, "Q-Former Adapter", fs=14, bold=True, color="#92640a")
txt(ACX + 2.5, AY + AH/2 - 0.30, "(trained)", fs=10, italic=True, color=ADAPT_EC)

# ── source_proj inside adapter (top, left-ish) ─────────────────────────────────
rbox(LX, 10.72, 3.6, 0.38, "#fffdf5", ADAPT_EC, lw=1.0, r=0.07, z=4)
txt(LX, 10.72, "source_proj :  2048 → 512", fs=9.5, color="#6b4a00")

arr(LX, 10.53, LX, 10.15, color=ADAPT_EC, lw=1.3)  # down into cross-attn

# ── LEARNED QUERIES  (left side, prominent) ────────────────────────────────────
QX, QY = 2.35, 9.1
rbox(QX, QY, 2.6, 1.55, QUERY_FC, QUERY_EC, lw=2.0, r=0.14, z=5)
txt(QX, QY + 0.48, "Learned Queries", fs=11, bold=True, color="#7a2e00")
txt(QX, QY + 0.08, "768 × 512", fs=10, color="#7a2e00", bold=True)
txt(QX, QY - 0.32, "trainable parameters", fs=8.5, italic=True, color="#a04010")
# star symbol to indicate "learned"
txt(QX + 1.15, QY + 0.48, "★", fs=13, color=QUERY_EC)

# ── ATTENTION BLOCK  (centre) ──────────────────────────────────────────────────
BX, BY = 7.2, 9.1
BW, BH = 5.0, 1.70
rbox(BX, BY, BW, BH, BLOCK_FC, BLOCK_EC, lw=1.5, r=0.12, z=4)

txt(BX, BY + 0.62, "①  Self-Attention", fs=10, bold=True, color="#5a3e00", ha="center")
txt(BX + 0.6, BY + 0.62, "  queries → queries", fs=9, color="#555", ha="left")

txt(BX, BY + 0.15, "②  Cross-Attention", fs=10, bold=True, color="#5a3e00", ha="center")
txt(BX + 0.6, BY + 0.15, "  queries ← source tokens", fs=9, color="#555", ha="left")

txt(BX, BY - 0.32, "③  Feed-Forward Network", fs=10, bold=True, color="#5a3e00", ha="center")

txt(BX + 2.2, BY - 0.70, "× 4 blocks", fs=9, italic=True, color="#888")

# arrows  queries → block,   source_proj → block
arr(QX + 1.3, QY, BX - BW/2, BY, color=QUERY_EC, lw=1.8)   # queries into block
arr(LX, 10.15, BX - BW/2, BY + 0.15, color=ADAPT_EC, lw=1.5)  # tokens into cross-attn

# ── out_proj ───────────────────────────────────────────────────────────────────
OPROJ_X, OPROJ_Y = ACX, 8.02
arr(BX, BY - BH/2, OPROJ_X, OPROJ_Y + 0.22, color=ADAPT_EC, lw=1.5)

rbox(OPROJ_X, OPROJ_Y, 3.8, 0.38, "#fffdf5", ADAPT_EC, lw=1.0, r=0.07, z=4)
txt(OPROJ_X, OPROJ_Y, "LayerNorm  +  out_proj :  512 → 128", fs=9.5, color="#6b4a00")

# ── z_pred out of adapter ──────────────────────────────────────────────────────
arr(ACX, AY - AH/2, ACX, 7.32)
tensor(ACX, 7.17, "(B, 768, 128)   —   z_pred")

# ═══════════════════════════════════════════════════════════════════════════════
# NOVA3R ODE DECODER
# ═══════════════════════════════════════════════════════════════════════════════
arr(ACX, 7.02, ACX, 6.47)
rbox(ACX, 6.10, 5.4, 0.72, FROZEN_FC, FROZEN_EC, lw=2.0)
txt(ACX, 6.28, "NOVA3R ODE Decoder", fs=12, bold=True, color="#1a4f7a")
txt(ACX, 5.93, "flow matching  ·  25 Euler steps  ·  frozen", fs=9, italic=True, color="#2e86c1")

arr(ACX, 5.74, ACX, 5.28)
tensor(ACX, 5.13, "8 192 predicted 3-D points")

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PATH  (NOVA3R encoder)    centre x = 11.5
# ═══════════════════════════════════════════════════════════════════════════════
RX = 11.5

rbox(RX, 15.4, 2.8, 0.75, INPUT_FC, INPUT_EC, lw=1.1, r=0.10)
txt(RX, 15.57, "GT Replica Mesh", fs=11, bold=True)
txt(RX, 15.22, "ground-truth geometry", fs=9, color="#555", italic=True)

arr(RX, 15.02, RX, 14.42)
rbox(RX, 14.05, 2.8, 0.75, "#fef9ef", "#c8922a", lw=1.5, r=0.10)
txt(RX, 14.23, "Frustum Clip", fs=11, bold=True, color="#7a4800")
txt(RX, 13.87, "union of 8 camera frustums", fs=9, italic=True, color="#c8922a")

arr(RX, 13.67, RX, 13.22)
tensor(RX, 13.07, "8 192 GT\nmesh points")

arr(RX, 12.92, RX, 12.32)
rbox(RX, 11.95, 2.8, 0.72, FROZEN_FC, FROZEN_EC, lw=2.0)
txt(RX, 12.13, "NOVA3R AE Encoder", fs=11, bold=True, color="#1a4f7a")
txt(RX, 11.77, "frozen", fs=9, italic=True, color="#2e86c1")

arr(RX, 11.59, RX, 11.17)
tensor(RX, 11.02, "(B, 768, 128)   —   z_star")
txt(RX, 10.72, "training target", fs=8.5, italic=True, color="#888")

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_X, LOSS_Y = 9.6, 7.17

rbox(LOSS_X, LOSS_Y, 1.8, 0.58, LOSS_FC, LOSS_EC, lw=2.0, r=0.12)
txt(LOSS_X, LOSS_Y + 0.08, "MSE Loss", fs=11, bold=True, color=LOSS_C)
txt(LOSS_X, LOSS_Y - 0.22, "gradients → adapter only", fs=8, italic=True, color=LOSS_C)

# z_pred → loss
darr(ACX + 1.2, 7.17, LOSS_X - 0.9, LOSS_Y)

# z_star → loss  (down from right column, then left)
darr(RX, 10.72, RX, LOSS_Y, rad=0.0)
darr(RX, LOSS_Y, LOSS_X + 0.9, LOSS_Y, rad=0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(fc=FROZEN_FC, ec=FROZEN_EC, lw=1.5, label="Frozen module"),
    mpatches.Patch(fc=ADAPT_FC,  ec=ADAPT_EC,  lw=1.5, label="Trained (adapter)"),
    mpatches.Patch(fc=QUERY_FC,  ec=QUERY_EC,  lw=1.5, label="Learned queries  ★"),
    mpatches.Patch(fc=TENSOR_FC, ec=TENSOR_EC, lw=1.0, label="Tensor / shape"),
    mpatches.Patch(fc=LOSS_FC,   ec=LOSS_EC,   lw=1.5, label="Loss  (dashed)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=10,
          framealpha=0.95, edgecolor="#cccccc",
          bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.2)
out = "experiments/overfit_8frames/architecture.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
