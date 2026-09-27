"""Render-only: rebuild the FM flow panel from the saved fm_states.npz (no model)."""
import argparse, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="/usr/prakt/s0016/vc3r/outputs/replica/fm_intermediates/fm_states.npz")
ap.add_argument("--elev", type=float, default=32.0)
ap.add_argument("--azim", type=float, default=-58.0)
ap.add_argument("--render-pts", type=int, default=15000)
ap.add_argument("--out", default="/usr/prakt/s0016/vc3r/outputs/replica/fm_intermediates/fm_flow_panel")
args = ap.parse_args()

d = np.load(args.npz)
clouds, ts, hn = d["clouds"], d["t"], d["hn"]
final = clouds[-1]
rng = np.random.default_rng(0)
ri = rng.choice(final.shape[0], min(args.render_pts, final.shape[0]), replace=False)
n = len(ts)

# shared limits from the final cloud; slightly tighter cube
lim = np.stack([final[ri].min(0), final[ri].max(0)])
ctr = lim.mean(0); rad = (lim[1] - lim[0]).max() * 0.55

fig = plt.figure(figsize=(2.55 * n, 2.9))
for j, tt in enumerate(ts):
    cw = clouds[j]
    ax = fig.add_subplot(1, n, j + 1, projection="3d")
    ax.scatter(cw[ri, 0], cw[ri, 1], cw[ri, 2], c=hn[ri], cmap="viridis",
               s=1.4, alpha=0.9, linewidths=0, vmin=0, vmax=1)
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    label = (r"$t=0$  (noise)" if tt == 0 else (r"$t=1$  (scene)" if tt >= 1 else fr"$t={tt:g}$"))
    ax.set_title(label, fontsize=12, pad=-6)
fig.suptitle(r"Flow-matching decode: uniform noise $\longrightarrow$ scene  (NOVA3R AE, Replica office4)",
             y=1.02, fontsize=13)
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.86, wspace=-0.12)
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig] {args.out}.png / .pdf")
