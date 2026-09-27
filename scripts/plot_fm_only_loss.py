"""Pure-FM adapter training curves (Replica, complete target, office4 val).
Shows the FM training loss saturating early (~4-6k) at its floor while the validation token-MSE
peaks around the first cosine cycle (6k) and then stalls -- i.e. the objective converges quickly
to a limited reconstruction. Plotted from the run-specific snapshot fm_only_complete_metrics.csv."""
import csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("/usr/prakt/s0016/vc3r/outputs/consecutive_windows/fm_only_complete_metrics.csv")
OUT = Path("/usr/prakt/s0016/vc3r/outputs/replica/fm_only_loss"); OUT.mkdir(parents=True, exist_ok=True)

step, tr, vm = [], [], []
for r in csv.DictReader(open(CSV)):
    step.append(int(r["step"])); tr.append(float(r["train_loss"])); vm.append(float(r["val_plain_mse"]))
step, tr, vm = np.array(step), np.array(tr), np.array(vm)

C_TR, C_VM, C_R = "#c05621", "#2b6cb0", "#718096"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False})
fig, ax = plt.subplots(figsize=(7.4, 4.4))

# left axis: FM training loss (log)
ax.set_yscale("log")
ax.plot(step, tr, color=C_TR, lw=1.3, alpha=0.5)
k = 5; sm = np.convolve(tr, np.ones(k)/k, mode="valid")
ax.plot(step[k-1:], sm, color=C_TR, lw=2.4, label="FM training loss (smoothed)")
ax.set_xlabel("training step"); ax.set_ylabel("FM velocity loss (log)", color=C_TR)
ax.tick_params(axis="y", labelcolor=C_TR)
ax.grid(True, which="both", alpha=0.15)

# right axis: validation token-MSE (linear)
ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(step, vm, color=C_VM, lw=2.2, label="val token-MSE (office4)")
ax2.set_ylabel("validation token-MSE", color=C_VM); ax2.tick_params(axis="y", labelcolor=C_VM)

# cosine restart + best-val markers
ax.axvline(6000, color=C_R, ls=(0, (4, 3)), lw=1.3)
ax.text(6100, ax.get_ylim()[1]*0.55, "cosine restart", color=C_R, fontsize=8.5, rotation=90, va="top")
jbest = int(np.argmin(vm)); ax2.scatter([step[jbest]], [vm[jbest]], color=C_VM, s=34, zorder=5)
ax2.annotate(f"best val ~{step[jbest]}", (step[jbest], vm[jbest]),
             textcoords="offset points", xytext=(6, 10), color=C_VM, fontsize=8.5)

# geometry annotation (from the office4 stitch eval)
ax.text(0.97, 0.93, "furniture F@2:  0.253 @4k  →  0.265 @18k\n"
        "on par with MSE at equal data (0.238); both\ndata-limited, below oracle 0.644",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7fafc", ec="#cbd5e0"))

l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, la1+la2, loc="lower left", fontsize=9, frameon=False)
ax.set_title("Flow matching as the adapter's sole training signal (Replica office4)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fm_only_loss.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {OUT}/fm_only_loss.png / .pdf  ({len(step)} points, best-val step {step[jbest]})")
