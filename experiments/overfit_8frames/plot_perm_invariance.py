#!/usr/bin/env python3
"""Two-panel permutation-invariance figure (AVERAGED over windows).

Reads perm_invariance_avg.json (16 windows, 2 per Replica room).
Left  (Claim 1 / decoder):  order-shuffle ~ 0, while a graded token-value
                            perturbation (sigma sweep) and full random climb up.
Right (Claim 2 / encoder):  identity ~ random >> Hungarian, error bars across
                            windows; both seeds decode to one geometry.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parent
d = json.loads((EXP / "perm_invariance_avg.json").read_text())
dec, enc, sig = d["decoder"], d["encoder"], d["noise_sigmas"]
N = d["n_windows"]
GREEN, RED, GREY, ORANGE = "#2e7d32", "#c0392b", "#95a5a6", "#e08e0b"


def mv(node, k):  # (mean, std)
    return node[k]["mean"], node[k]["std"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8))

# ---- Panel A: decoder — order vs graded value corruption -------------------
keys   = ["shuffle"] + [f"noise{s}" for s in sig] + ["random"]
labels = ["shuffle\nORDER"] + [f"+noise\n$\\sigma{{=}}{s}$" for s in sig] + ["full\nrandom"]
cols   = [GREEN] + [ORANGE] * len(sig) + [RED]
means  = [mv(dec, k)[0] for k in keys]
stds   = [mv(dec, k)[1] for k in keys]
xa = np.arange(len(keys))
axL.bar(xa, means, yerr=stds, width=0.62, color=cols, edgecolor="black",
        linewidth=0.6, capsize=3)
axL.set_yscale("log"); axL.set_ylim(1e-4, 2e0)
axL.set_xticks(xa); axL.set_xticklabels(labels, fontsize=8.5)
axL.set_ylabel("Chamfer to original decode\n(normalized units, log scale)")
axL.set_title("(a) Decoder ignores token ORDER, not VALUES",
              fontsize=10.5, fontweight="bold")
for x, m in zip(xa, means):
    axL.text(x, m * 1.5, (f"{m:.0e}" if m < 1e-2 else f"{m:.2f}"),
             ha="center", va="bottom", fontsize=8)
axL.text(0.02, 0.95, f"shuffle: per-point mean$|\\Delta|$ = {dec['ptdelta']['mean']:.0e}\n"
         f"(point-identical); noise $\\sigma$ = frac. of token std",
         transform=axL.transAxes, fontsize=7.5, color="black", va="top")

# ---- Panel B: encoder — arbitrary order (token space) + geometry bar -------
# Left axis: token distances (identity/random/hungarian). Right axis (twin):
# the decoded geometry barely moves across seeds -> decode seed-pair Chamfer.
bk = ["identity", "randperm", "hungarian"]
blab = ["identity\n(slot s vs s)", "random\nperm.", "Hungarian\n(optimal)"]
bm = [mv(enc, k)[0] for k in bk]; bs = [mv(enc, k)[1] for k in bk]
xb = np.arange(3)
axR.bar(xb, bm, yerr=bs, width=0.6, color=[GREY, GREY, GREEN],
        edgecolor="black", linewidth=0.6, capsize=4)
axR.set_ylabel("token distance $\\|z_i - z_j\\|$  (set space)")
axR.set_title("(b) Encoder emits the set in arbitrary ORDER",
              fontsize=10.5, fontweight="bold")
axR.set_ylim(0, 22)
for x, m, sd in zip(xb, bm, bs):
    axR.text(x, m + sd + 0.5, f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)
axR.annotate("", xy=(1.0, 19.6), xytext=(0.0, 19.6),
             arrowprops=dict(arrowstyle="<->", color=GREY, lw=1))
axR.text(0.5, 20.0, "identity $\\approx$ random", ha="center", fontsize=8, color=GREY)
axR.text(2.0, bm[2] + 2.6, f"{bm[0]/bm[2]:.1f}$\\times$", ha="center", fontsize=9, color=GREEN)

# geometry bar on a secondary axis, set apart from the token bars
BLUE = "#2b6cb0"
axR2 = axR.twinx()
xg = 3.6
gm, gs = mv(enc, "decode_seedvar")
axR2.bar([xg], [gm], yerr=[gs], width=0.6, color=BLUE, edgecolor="black",
         linewidth=0.6, capsize=4)
axR2.set_ylim(0, 1.0)
axR2.set_ylabel("decode Chamfer (norm. units)", color=BLUE)
axR2.tick_params(axis="y", colors=BLUE)
axR2.text(xg, gm + gs + 0.02, f"{gm:.1e}", ha="center", va="bottom", fontsize=8.5, color=BLUE)
axR2.text(xg, 0.30, "decode\nis identical\nacross seeds", ha="center", va="bottom",
          fontsize=8, color=BLUE)
axR.axvline(2.8, color="0.8", lw=0.8, ls="--")
axR.set_xlim(-0.6, 4.2)
axR.set_xticks(list(xb) + [xg])
axR.set_xticklabels(blab + ["decode\n(diff. seed)"], fontsize=8.5)

fig.suptitle(f"The NOVA3R latent is a set (mean$\\pm$std over {N} windows): "
             f"order-invariant decoding (a), arbitrarily-ordered encoding (b)",
             fontsize=10, y=1.03)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(EXP / f"perm_invariance.{ext}", bbox_inches="tight", dpi=200)
print("wrote perm_invariance.pdf / .png")
