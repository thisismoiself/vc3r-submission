#!/usr/bin/env python3
"""Per-window (single-prediction) eval from a stitch run's per_window.npz — scores each window's
adapter prediction against THAT window's own frustum GT (the input pool), so no multi-window
stitching/integration error is folded in. Saves each single-window pred/oracle/gt cloud. CPU only."""
import sys, argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def eval_cloud(pred, gt, n=200000, thr=(0.01, 0.05)):
    if len(pred) == 0 or len(gt) == 0:
        return {"chamfer_m": float("nan"), **{f"F@{int(t*100)}cm": 0.0 for t in thr}}
    rng = np.random.default_rng(0)
    p = pred if len(pred) <= n else pred[rng.choice(len(pred), n, replace=False)]
    g = gt if len(gt) <= n else gt[rng.choice(len(gt), n, replace=False)]
    d_pg = cKDTree(g).query(p, k=1)[0]        # pred->GT (accuracy)
    d_gp = cKDTree(p).query(g, k=1)[0]        # GT->pred (completeness)
    out = {"accuracy_m": float(d_pg.mean()), "completeness_m": float(d_gp.mean()),
           "chamfer_m": float((d_pg.mean() + d_gp.mean()) / 2)}
    for t in thr:
        prec = float((d_pg < t).mean()); rec = float((d_gp < t).mean())
        out[f"F@{int(t*100)}cm"] = 2 * prec * rec / (prec + rec + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-dir", required=True)
    ap.add_argument("--save-clouds", action="store_true")
    args = ap.parse_args()
    d = Path(args.stitch_dir)
    z = np.load(d / "per_window.npz")
    pred, orac, inp = z["pred"], z["oracle"], z["input"]     # [W, Q, 3]
    f0, f1 = z["f0"], z["f1"]
    W = pred.shape[0]
    print(f"{W} windows, {pred.shape[1]} pts each. Per-window pred vs its OWN frustum GT (no stitch integration):\n")
    print(f"{'win':>3} {'frames':>10} | {'PRED chamfer':>12} {'F@1':>5} {'F@5':>5} | {'ORACLE chamfer':>14} {'F@1':>5} {'F@5':>5}")
    agg_p, agg_o = [], []
    for w in range(W):
        gt = inp[w]                                          # this window's frustum GT (input pool)
        mp = eval_cloud(pred[w], gt); mo = eval_cloud(orac[w], gt)
        agg_p.append(mp); agg_o.append(mo)
        print(f"{w:3d} {f0[w]:4d}-{f1[w]:4d} | {mp['chamfer_m']*100:9.2f}cm {mp['F@1cm']:.3f} {mp['F@5cm']:.3f} | "
              f"{mo['chamfer_m']*100:11.2f}cm {mo['F@1cm']:.3f} {mo['F@5cm']:.3f}")
        if args.save_clouds:
            trimesh.PointCloud(pred[w]).export(d / f"win{w:02d}_pred.ply")
            trimesh.PointCloud(orac[w]).export(d / f"win{w:02d}_oracle.ply")
            trimesh.PointCloud(gt).export(d / f"win{w:02d}_gt.ply")
    mean = lambda a, k: float(np.mean([x[k] for x in a]))
    print("-" * 78)
    print(f"MEAN     PRED chamfer {mean(agg_p,'chamfer_m')*100:.2f}cm  F@1 {mean(agg_p,'F@1cm'):.3f}  F@5 {mean(agg_p,'F@5cm'):.3f}   |   "
          f"ORACLE chamfer {mean(agg_o,'chamfer_m')*100:.2f}cm  F@5 {mean(agg_o,'F@5cm'):.3f}")


if __name__ == "__main__":
    main()
