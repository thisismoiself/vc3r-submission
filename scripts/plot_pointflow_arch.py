"""Architecture diagram for the point-flow corrector (train_point_flow.py).
Top: deployment pipeline (frozen adapter + frozen decoder + trained FM corrector).
Bottom: the corrector's internal velocity field v(x_t, t | z_pred)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/usr/prakt/s0016/vc3r/outputs/replica/pointflow_arch")
OUT.mkdir(parents=True, exist_ok=True)

FROZEN = "#cbd5e0"; FROZEN_E = "#718096"      # gray  = frozen (reused, not trained)
TRAIN  = "#c6f6d5"; TRAIN_E  = "#2f855a"      # green = the only trained module
DATA   = "#ebf3fb"; DATA_E   = "#2b6cb0"      # blue  = tensors / clouds
GT     = "#fed7d7"; GT_E     = "#c53030"      # red   = GT (training only)

def box(ax, x, y, w, h, text, fc, ec, fs=10, lw=1.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=3)

def arrow(ax, x0, y0, x1, y1, ec="#2d3748", lw=1.7, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                 lw=lw, color=ec, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))

fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 8.2),
                               gridspec_kw={"height_ratios": [1, 1.25]})
for ax in (axT, axB):
    ax.set_xlim(0, 12); ax.axis("off")

# ---------------------------------------------------------------- TOP: pipeline
axT.set_ylim(0, 3.2)
axT.text(0.05, 3.02, "A.  Deployment pipeline — only the green module is trained",
         fontsize=12.5, weight="bold")
y = 1.5; h = 0.9
box(axT, 0.15, y, 1.55, h, "DA3\ntokens\n[N,2048]", DATA, DATA_E, 9)
box(axT, 2.05, y, 1.9, h, "Adapter\n(cross-attn)\n768×128", FROZEN, FROZEN_E, 9.5)
box(axT, 4.30, y, 1.35, h, r"$z_{pred}$" + "\ntokens\n768×128", DATA, DATA_E, 9)
box(axT, 6.00, y, 1.9, h, "NOVA3R\ndecoder (FM)", FROZEN, FROZEN_E, 9.5)
box(axT, 8.25, y, 1.45, h, r"$P_{pred}$" + "\ncloud\n[N,3]", DATA, DATA_E, 9)
box(axT, 10.05, y, 1.75, h, "FM\ncorrector\n(midpoint ODE)", TRAIN, TRAIN_E, 9.5, lw=2.4)
for x0, x1 in [(1.70, 2.05), (3.95, 4.30), (5.65, 6.00), (7.90, 8.25), (9.70, 10.05)]:
    arrow(axT, x0, y + h/2, x1, y + h/2)
# corrected output below the corrector
box(axT, 10.05, 0.15, 1.75, 0.72, "corrected\ncloud → GT surf.", DATA, DATA_E, 9)
arrow(axT, 10.9, y, 10.9, 0.87)
# frozen tags
for x, lbl in [(3.0, "frozen"), (6.95, "frozen")]:
    axT.text(x, y - 0.28, "❄ " + lbl, ha="center", fontsize=8.5, color=FROZEN_E)
axT.text(10.9, y + h + 0.22, "trained", ha="center", fontsize=9, color=TRAIN_E, weight="bold")
# z_pred also conditions the corrector (dashed skip)
arrow(axT, 4.97, y + h, 10.5, y + h + 0.42, ec=TRAIN_E, lw=1.5, ls=(0, (4, 3)), rad=-0.16)
axT.text(7.35, y + h + 0.30, r"conditions on $z_{pred}$ (GT-free at deploy)",
         ha="center", fontsize=8.5, color=TRAIN_E,
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# --------------------------------------------------------- BOTTOM: corrector net
axB.set_ylim(0, 3.7)
axB.text(0.05, 3.5, r"B.  FM corrector — velocity field $v_\theta(x_t,\,t \mid z_{pred})$   "
         "(4.3M params, 4× cross-attn, hidden 256)", fontsize=12.5, weight="bold")
# inputs
box(axB, 0.15, 2.55, 1.75, 0.7, r"point $x_t$ [B,N,3]", DATA, DATA_E, 9)
box(axB, 0.15, 1.55, 1.75, 0.7, r"time $t$ [B,N]", DATA, DATA_E, 9)
box(axB, 0.15, 0.35, 1.75, 0.7, r"$z_{pred}$ 768×128", DATA, DATA_E, 9)
# embeddings
box(axB, 2.35, 2.55, 2.1, 0.7, "Fourier(8) →\nLinear 48→256", "#fff", "#4a5568", 8.6)
box(axB, 2.35, 1.55, 2.1, 0.7, "Fourier(64) →\nLinear 128→256", "#fff", "#4a5568", 8.6)
box(axB, 2.35, 0.35, 2.1, 0.7, "Linear 128→256\n(token memory)", "#fff", "#4a5568", 8.6)
arrow(axB, 1.90, 2.90, 2.35, 2.90); arrow(axB, 1.90, 1.90, 2.35, 1.90); arrow(axB, 1.90, 0.70, 2.35, 0.70)
# sum of point+time
box(axB, 4.95, 2.05, 0.7, 1.2, "+", "#fff", "#4a5568", 17)
arrow(axB, 4.45, 2.90, 4.95, 2.75); arrow(axB, 4.45, 1.90, 4.95, 2.05 + 0.3)
axB.text(5.30, 3.32, "point ⊕ time", ha="center", fontsize=8, color="#4a5568")
# cross-attn stack
box(axB, 6.15, 1.55, 2.55, 1.7, "4×  CrossAttentionBlock\n(dim 256, 8 heads)\n"
    "queries=points\nmemory=tokens", TRAIN, TRAIN_E, 9.2, lw=2.2)
arrow(axB, 5.65, 2.65, 6.15, 2.55)                       # point+time -> blocks (as queries)
arrow(axB, 4.45, 0.70, 6.0, 1.75, rad=-0.12)              # token memory -> blocks
axB.text(5.1, 1.15, "memory", fontsize=8, color=TRAIN_E)
# head + output
box(axB, 9.05, 1.9, 1.55, 1.0, "LayerNorm\nLinear 256→3\n(×0.1 init)", "#fff", "#4a5568", 8.6)
arrow(axB, 8.70, 2.4, 9.05, 2.4)
box(axB, 10.9, 1.9, 0.95, 1.0, r"$v_\theta$" + "\n[B,N,3]", DATA, DATA_E, 9)
arrow(axB, 10.60, 2.4, 10.9, 2.4)

# FM training recipe box
axB.text(6.15, 0.98, "Training (NN-coupled straight CFM):", fontsize=9, color=TRAIN_E, weight="bold")
axB.text(6.15, 0.10,
         r"$x_0\!\sim\!P_{pred}$,  $x_1=$nearest GT surf. pt,  "
         r"$x_t=(1{-}t)x_0+t\,x_1$,  $v^\star=x_1-x_0$" + "\n"
         r"loss $=\| v_\theta(x_t,t\mid z_{pred}) - v^\star\|^2$ ;  "
         r"deploy: integrate $P_{pred}$ forward (midpoint, few steps)",
         fontsize=8.7, va="bottom")

fig.suptitle("Point-flow corrector: post-hoc flow-matching refinement in 3D point space",
             fontsize=13.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/pointflow_arch.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {OUT}/pointflow_arch.png / .pdf")
