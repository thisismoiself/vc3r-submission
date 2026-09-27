#!/usr/bin/env python3
"""Offline (CPU) re-scoring of a saved whole-room udf_pred under different output post-filters:
baseline vs frontier-mask (restrict prediction to unknown near observed) vs +connected-components
(keep only prediction connected to observed structure). No GPU / no re-sampling needed."""
import sys, argparse
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, label
OBSERVED, FREE, UNKNOWN = 0, 1, 2


def score(udf_pred, mask, gt, pt, voxel, band, frontier=0, cc=False):
    surf = 0.7 * voxel
    obs = (mask == OBSERVED) & (np.abs(pt) < surf)
    gt_new = (mask == UNKNOWN) & (np.clip(gt, 0, band) < surf)
    true_surf = obs | gt_new
    unk = (mask == UNKNOWN); n_gt = int(gt_new.sum())
    cand = unk & binary_dilation(obs, iterations=frontier) if frontier > 0 else unk
    pred = np.zeros(mask.shape, bool); nc = int(cand.sum())
    if n_gt and nc:
        k = min(n_gt, nc); thr = np.partition(udf_pred[cand], k - 1)[k - 1]
        pred = cand & (udf_pred <= thr)
    if cc and pred.any():
        lbl, _ = label(pred | obs); pred = pred & np.isin(lbl, np.unique(lbl[obs]))
    dist_true = distance_transform_edt(~true_surf)
    dist_pred = distance_transform_edt(~pred) if pred.any() else np.full(mask.shape, 1e6, np.float32)
    recall = 100 * float(np.mean(dist_pred[gt_new] <= 1.0)) if n_gt else 0.0
    dp = dist_true[pred]
    prec = 100 * float(np.mean(dp <= 1.0)) if pred.any() else 0.0
    halluc = 100 * float(np.mean(dp > 3.0)) if pred.any() else 0.0
    F = 2 * prec * recall / max(1e-9, prec + recall)
    return F, recall, prec, halluc, int(pred.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)         # the DA3 npz (mask/gt/partial)
    ap.add_argument("--udfpred", required=True)        # saved udf_pred npz
    args = ap.parse_args()
    d = np.load(args.cache); mask = d["mask"]; gt = np.abs(d["gt_tsdf"].astype(np.float32))
    pt = np.abs(d["partial_tsdf"].astype(np.float32)); voxel = float(d["voxel"])
    band = float(d["band"]) if "band" in d else 0.10
    udf = np.load(args.udfpred)["udf_pred"].astype(np.float32)
    print(f"{'config':28s}  F      recall  prec    halluc  npred")
    for name, fr, cc in [("baseline", 0, False), ("frontier-mask(10)", 10, False),
                         ("frontier-mask(15)", 15, False), ("frontier(15)+cc", 15, True),
                         ("cc-only", 0, True)]:
        F, r, p, h, n = score(udf, mask, gt, pt, voxel, band, fr, cc)
        print(f"{name:28s}  {F:5.1f}  {r:5.1f}%  {p:5.1f}%  {h:5.1f}%  {n:,}")


if __name__ == "__main__":
    main()
