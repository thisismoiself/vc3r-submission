#!/usr/bin/env python3
"""DA3-RANSAC drop-k re-eval on an existing office4 stitch.

Reuses the per-window PRED clouds already decoded by stitch_office4.py (world frame,
saved in per_window.npz) and the DA3 outlier ranking from da3_ref_test.py (da3_ref.npz).
Re-stitches with the worst-k windows removed and re-computes the whole-scene and
furniture-region metrics with the *identical* eval_cloud + furniture crop as
stitch_office4.py, so numbers are directly comparable.

Three keep-sets are reported:
  all       : every window (baseline, reproduces the stitch report)
  drop-k    : drop the k windows with the largest pred->DA3 distance (GT-FREE, deployable)
  drop-k*   : drop the k windows with the largest true pred->GT error (GT ceiling; how much
              the GT-free ranking leaves on the table)
"""
import argparse, json
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

REPO = Path("/usr/prakt/s0016/vc3r")


def eval_cloud(pred, gt_tree, gt_pts, n=200000, thresholds=(0.01, 0.05)):
    """Identical to stitch_office4.eval_cloud."""
    if len(pred) == 0 or len(gt_pts) == 0:
        out = {k: float("nan") for k in
               ["accuracy_m", "completeness_m", "chamfer_m", "acc_median_m", "comp_median_m"]}
        for t in thresholds:
            out[f"F@{int(t*100)}cm"] = out[f"prec@{int(t*100)}cm"] = out[f"recall@{int(t*100)}cm"] = 0.0
        return out
    rng = np.random.default_rng(0)
    p = pred if len(pred) <= n else pred[rng.choice(len(pred), n, replace=False)]
    g = gt_pts if len(gt_pts) <= n else gt_pts[rng.choice(len(gt_pts), n, replace=False)]
    d_pg, _ = gt_tree.query(p, k=1)              # pred -> GT (full GT tree; accuracy)
    pred_tree = cKDTree(p)
    d_gp, _ = pred_tree.query(g, k=1)            # GT -> pred (completeness)
    acc, comp = float(d_pg.mean()), float(d_gp.mean())
    out = {"accuracy_m": acc, "completeness_m": comp, "chamfer_m": (acc + comp) / 2,
           "acc_median_m": float(np.median(d_pg)), "comp_median_m": float(np.median(d_gp))}
    for t in thresholds:
        prec = float((d_pg < t).mean()); rec = float((d_gp < t).mean())
        f = 2 * prec * rec / (prec + rec + 1e-9)
        out[f"F@{int(t*100)}cm"] = f
        out[f"prec@{int(t*100)}cm"] = prec
        out[f"recall@{int(t*100)}cm"] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitch-dir", required=True)
    ap.add_argument("--room", default="office4")
    ap.add_argument("--drop-k", type=int, default=3)
    args = ap.parse_args()
    sd = Path(args.stitch_dir)

    z = np.load(sd / "per_window.npz")
    pred_w = z["pred"]                                   # (W, num_q, 3) world frame
    W = pred_w.shape[0]
    ref = np.load(sd / "da3_ref.npz")
    pred_da3 = ref["pred_da3"]                           # GT-free outlier signal
    err_gt = ref["err"]                                  # true pred->GT error (ceiling)

    gt = np.asarray(trimesh.load(
        str(REPO / "outputs/replica/gt_pointclouds" / args.room / f"{args.room}_gt_2m.ply")).vertices)
    gt_tree = cKDTree(gt)

    # furniture crop bounds — identical derivation to stitch_office4.py
    zc = gt[:, 2]; floor = float(np.percentile(zc, 1))
    xy_lo = np.percentile(gt[:, :2], 1, axis=0); xy_hi = np.percentile(gt[:, :2], 99, axis=0)
    def fmask(p, margin=0.4, zlo=0.15, zhi=1.1):
        zz = p[:, 2]
        return ((zz > floor + zlo) & (zz < floor + zhi)
                & (p[:, 0] > xy_lo[0] + margin) & (p[:, 0] < xy_hi[0] - margin)
                & (p[:, 1] > xy_lo[1] + margin) & (p[:, 1] < xy_hi[1] - margin))
    gt_furn = gt[fmask(gt)]; gt_furn_tree = cKDTree(gt_furn)

    def stitch_eval(keep):
        cloud = np.concatenate([pred_w[i] for i in keep], 0)
        whole = eval_cloud(cloud, gt_tree, gt)
        furn = eval_cloud(cloud[fmask(cloud)], gt_furn_tree, gt_furn, thresholds=(0.02, 0.05))
        return whole, furn

    all_idx = list(range(W))
    worst_free = [int(i) for i in np.argsort(pred_da3)[::-1][:args.drop_k]]
    worst_ceil = [int(i) for i in np.argsort(err_gt)[::-1][:args.drop_k]]
    drop_free = sorted(set(all_idx) - set(worst_free))
    drop_ceil = sorted(set(all_idx) - set(worst_ceil))
    dropped_free = sorted(worst_free)
    dropped_ceil = sorted(worst_ceil)

    configs = [("all", all_idx, []),
               (f"drop-{args.drop_k} (GT-free)", drop_free, dropped_free),
               (f"drop-{args.drop_k}* (GT ceiling)", drop_ceil, dropped_ceil)]

    print(f"\n{'config':22s} {'wChamfer':>9s} {'wF@5':>7s} {'FURN F@2':>9s} {'FURN F@5':>9s}   dropped")
    print("-" * 78)
    out = {}
    for name, keep, dropped in configs:
        whole, furn = stitch_eval(keep)
        out[name] = {"whole": whole, "furn": furn, "dropped": dropped}
        print(f"{name:22s} {whole['chamfer_m']:9.4f} {whole['F@5cm']:7.3f} "
              f"{furn['F@2cm']:9.3f} {furn['F@5cm']:9.3f}   {dropped}")
    (sd / "ransac_drop_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {sd/'ransac_drop_eval.json'}")


if __name__ == "__main__":
    main()
