"""Decoder token-noise sensitivity (complete-target). How much does the frozen NOVA3R decoder's
output move when we perturb the true tokens z* by isotropic Gaussian noise of a given per-element
MSE (same units as the adapter's val_plain_mse)? Reference lines: the irreducible sampler floor
(cross-seed decode scatter) and the adapter's real, STRUCTURED error (pred->oracle). Explanatory
notes sit in a box at the bottom. Data: experiments/overfit_8frames/token_sensitivity_complete.csv"""
import csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("/usr/prakt/s0016/vc3r/experiments/overfit_8frames/token_sensitivity_complete.csv")
OUT = Path("/usr/prakt/s0016/vc3r/outputs/replica/token_sensitivity"); OUT.mkdir(parents=True, exist_ok=True)

FLOOR   = 2.54     # cm, cross-seed decode scatter (irreducible)
ADPT_MSE = 0.073   # complete MSE-1225 adapter val_plain_mse
REAL_CM  = 5.19    # office4 real adapter pred->oracle Chamfer (structured token error)

mse, dev = [], []
for r in csv.DictReader(open(CSV)):
    if float(r["mse"]) > 0:
        mse.append(float(r["mse"])); dev.append(float(r["dev_cm"]))
mse, dev = np.array(mse), np.array(dev)
iso_at_adpt = float(np.interp(ADPT_MSE, mse, dev))      # isotropic damage at the adapter's MSE
factor = REAL_CM / iso_at_adpt

C_ISO, C_FLOOR, C_REAL, C_OP = "#2b6cb0", "#718096", "#c05621", "#2f855a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.6, 5.2))
fig.subplots_adjust(bottom=0.30)   # room for the notes box

ax.plot(mse, dev, "-o", color=C_ISO, lw=2, ms=5, label="isotropic token noise")
ax.axhline(FLOOR, color=C_FLOOR, ls=(0, (5, 3)), lw=1.6, label=f"sampler floor {FLOOR:.2f} cm (irreducible)")
ax.axvline(ADPT_MSE, color=C_OP, ls=(0, (4, 3)), lw=1.5)
ax.scatter([ADPT_MSE], [iso_at_adpt], color=C_ISO, s=55, zorder=6, edgecolor="white")
ax.scatter([ADPT_MSE], [REAL_CM], color=C_REAL, marker="*", s=260, zorder=6, edgecolor="white",
           label=f"real adapter (structured) {REAL_CM:.2f} cm")
ax.annotate("", xy=(ADPT_MSE, REAL_CM), xytext=(ADPT_MSE, iso_at_adpt),
            arrowprops=dict(arrowstyle="<->", color=C_REAL, lw=1.4))
ax.text(ADPT_MSE*1.12, (REAL_CM+iso_at_adpt)/2, fr"$\times{factor:.2f}$", color=C_REAL, fontsize=11, va="center")
ax.set_xscale("log")
ax.set_xlabel("token per-element MSE")
ax.set_ylabel("decode deviation from clean z* (cm)")
ax.set_title("Frozen decoder: sensitivity to token noise (complete target)")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(True, which="both", alpha=0.15)
ax.text(ADPT_MSE*1.1, ax.get_ylim()[1]*0.80, "adapter\noperates here", color=C_OP,
        fontsize=8.5, ha="left", va="top")

notes = (
    r"$\bullet$ Adapter token-MSE $\approx 0.073$  (its operating point, green line)." "\n"
    fr"$\bullet$ Isotropic noise at this MSE $\to$ {iso_at_adpt:.1f} cm deviation — barely above the {FLOOR:.2f} cm sampler floor." "\n"
    fr"$\bullet$ Real (structured) adapter error is {REAL_CM:.2f} cm $= \times{factor:.2f}$ the isotropic-equivalent — a modest structure penalty, no blow-up." "\n"
    r"$\Rightarrow$ Decoder is well-conditioned: robust to small token noise, degrades smoothly; residual error is the adapter, not decoder fragility."
)
fig.text(0.06, 0.015, notes, fontsize=9, va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f7fafc", ec="#cbd5e0"))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/token_sensitivity.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {OUT}/token_sensitivity.png / .pdf  | iso@adpt={iso_at_adpt:.2f}cm factor=x{factor:.2f}")
