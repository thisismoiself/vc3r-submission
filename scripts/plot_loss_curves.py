"""Loss curves for the three complete-target runs used in the presentation:
  (A) Replica adapter        -> fullrec_nf16_vel03_complete.log   (plain-MSE, 18k steps)
  (B) +NeuralRGBD adapter    -> fullrec_nrgbd_complete.log        (plain-MSE, 18k steps)
  (C) Point-flow corrector   -> point_flow_complete.log           (OT flow loss, 8k steps)
Panels A/B share the MSE axis (directly comparable); C is on its own axis."""
import re, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = Path("/usr/prakt/s0016/vc3r/experiments/overfit_8frames")
OUT = Path("/usr/prakt/s0016/vc3r/outputs/replica/loss_curves")
OUT.mkdir(parents=True, exist_ok=True)

def parse_adapter(fn):
    """step / train plain-MSE / val plain-MSE from the fullrec logs."""
    pat = re.compile(r"step\s+(\d+)/\d+.*?train=([\d.]+).*?val_mse=([\d.]+)")
    s, tr, va = [], [], []
    for ln in open(LOGDIR / fn):
        m = pat.search(ln)
        if m:
            s.append(int(m.group(1))); tr.append(float(m.group(2))); va.append(float(m.group(3)))
    return np.array(s), np.array(tr), np.array(va)

def parse_pointflow(fn):
    pat = re.compile(r"\[train\] step (\d+)/\d+\s+ot_flow_loss=([\d.]+)")
    s, l = [], []
    for ln in open(LOGDIR / fn):
        m = pat.search(ln)
        if m:
            s.append(int(m.group(1))); l.append(float(m.group(2)))
    return np.array(s), np.array(l)

r_s, r_tr, r_va = parse_adapter("fullrec_nf16_vel03_complete.log")
n_s, n_tr, n_va = parse_adapter("fullrec_nrgbd_complete.log")
p_s, p_l = parse_pointflow("point_flow_complete.log")

C_REP, C_NRG, C_PF = "#2b6cb0", "#c05621", "#2f855a"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: adapter plain-MSE, Replica vs +NRGBD (train solid, val dashed)
ax0.plot(r_s, r_tr, color=C_REP, lw=2, label="Replica  train")
ax0.plot(r_s, r_va, color=C_REP, lw=1.8, ls="--", label="Replica  val")
ax0.plot(n_s, n_tr, color=C_NRG, lw=2, label="+NeuralRGBD  train")
ax0.plot(n_s, n_va, color=C_NRG, lw=1.8, ls="--", label="+NeuralRGBD  val")
ax0.set_yscale("log")
ax0.set_xlabel("training step"); ax0.set_ylabel("plain Hungarian MSE (log)")
ax0.set_title("Adapter training loss")
ax0.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")

# Panel B: point-flow corrector OT flow loss
ax1.plot(p_s, p_l, color=C_PF, lw=1.4, alpha=0.55, label="_raw")
# running mean to read the trend through the per-step noise
if len(p_l) >= 5:
    k = 5; sm = np.convolve(p_l, np.ones(k) / k, mode="valid")
    ax1.plot(p_s[k - 1:], sm, color=C_PF, lw=2.4, label="OT flow loss (smoothed)")
ax1.set_yscale("log")
ax1.set_xlabel("training step"); ax1.set_ylabel("OT flow-matching loss (log)")
ax1.set_title("Point-flow corrector")
ax1.legend(frameon=False, fontsize=9, loc="upper right")

fig.suptitle("Complete-target training curves", fontsize=13, y=1.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/loss_curves.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {OUT}/loss_curves.png / .pdf")
print(f"  Replica    : {len(r_s)} pts, final train_mse={r_tr[-1]:.4f} val_mse={r_va[-1]:.4f}")
print(f"  +NeuralRGBD: {len(n_s)} pts, final train_mse={n_tr[-1]:.4f} val_mse={n_va[-1]:.4f}")
print(f"  Point-flow : {len(p_s)} pts, final ot_flow={p_l[-1]:.5f}")
