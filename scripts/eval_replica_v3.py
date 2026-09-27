#!/usr/bin/env python3
"""Publication-style benchmark for the DA3 -> DiffComplete(v3) pipeline on Replica.
Per room, against the clean-mesh GT surface, for TWO outputs:
  * DA3-alone      : the raw DA3 reconstruction (the baseline we improve on)
  * v3 + bbox-clip : DA3 observed  UNION  v3 hole-fills (unknown & udf<thr), clipped to the room bbox
Metrics: Chamfer(cm, both-dir mean), F-score@1cm/@5cm (precision=accuracy, recall=completeness),
IoU(voxel occupancy), plus deep(>50cm)/shallow hole recall and exterior%.
"""
import sys, argparse, glob
import numpy as np
from scipy.spatial import cKDTree
OBS, FREE, UNK = 0, 1, 2


def pts_from(sel, origin, voxel):
    return origin + np.argwhere(sel) * voxel


def _room_frame(ref_xy):
    """Room orientation from the MINIMUM-AREA oriented rectangle of the observed floor footprint
    (rotating-calipers brute force over 0-90deg on the 2D convex hull). Robust where PCA fails: for a
    near-square/axis-aligned room PCA picks a spurious diagonal (OBB ends up BIGGER than the AABB); the
    min-area rect always returns the tightest box (~axis-aligned for office4, rotated for office2/room1).
    Returns (center, R) with R's columns the room axes."""
    ctr = ref_xy.mean(0)
    try:
        from scipy.spatial import ConvexHull
        pts = (ref_xy - ctr)[ConvexHull(ref_xy - ctr).vertices]
    except Exception:
        pts = ref_xy - ctr
    best = (float("inf"), 0.0)
    for a in np.deg2rad(np.arange(0, 90, 0.5)):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = pts @ R
        area = float(np.prod(q.max(0) - q.min(0)))
        if area < best[0]:
            best = (area, a)
    a = best[1]
    return ctr, np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def obb_clip(pts, ref, pad=0.05):
    """Clip `pts` to the min-area ORIENTED bounding box of `ref` (the observed room), Z world-vertical."""
    ctr, V = _room_frame(ref[:, :2])
    def to_room(P):
        Q = P.copy(); Q[:, :2] = (P[:, :2] - ctr) @ V; return Q
    rr = to_room(ref); lo, hi = rr.min(0) - pad, rr.max(0) + pad
    pr = to_room(pts)
    return pts[((pr >= lo) & (pr <= hi)).all(1)]


def chamfer_cm(A, B):
    dab = cKDTree(B).query(A, workers=4)[0]
    dba = cKDTree(A).query(B, workers=4)[0]
    return 100 * 0.5 * (dab.mean() + dba.mean())


def fscore(pred, gt, tau):
    prec = np.mean(cKDTree(gt).query(pred, workers=4)[0] <= tau)   # accuracy
    rec = np.mean(cKDTree(pred).query(gt, workers=4)[0] <= tau)    # completeness
    f = 2 * prec * rec / max(1e-9, prec + rec)
    return 100 * f, 100 * prec, 100 * rec


def iou_vox(pred, gt, origin, voxel):
    def key(P): return set(map(tuple, np.floor((P - origin) / voxel).astype(int)))
    a, b = key(pred), key(gt)
    return 100 * len(a & b) / max(1, len(a | b))


def eval_room(room, cache_dir, stitch_dir, thr=0.03):
    c = np.load(f"{cache_dir}/replica_{room}.npz")
    u = np.load(f"{stitch_dir}/replica_{room}_udfpred.npz")["udf_pred"].astype(np.float32)
    mask = c["mask"]; gt = np.abs(c["gt_tsdf"].astype(np.float32)); pt = np.abs(c["partial_tsdf"].astype(np.float32))
    origin = c["origin"].astype(np.float64); voxel = float(c["voxel"]); surf = 0.7 * voxel

    gt_pts = pts_from(gt < surf, origin, voxel)
    da3_pts = pts_from((mask == OBS) & (pt < surf), origin, voxel)
    fill_w = pts_from((mask == UNK) & (u < thr), origin, voxel)
    fill_w = obb_clip(fill_w, da3_pts)                                    # ORIENTED bbox clip
    comp_pts = np.concatenate([da3_pts, fill_w])

    # hole analysis (recall of occluded GT by the completion)
    d_gt_da3 = cKDTree(da3_pts).query(gt_pts, workers=4)[0]
    shallow = gt_pts[(d_gt_da3 > 3 * voxel) & (d_gt_da3 <= 25 * voxel)]
    deep = gt_pts[d_gt_da3 > 25 * voxel]
    def rec(H, P): return 100 * np.mean(cKDTree(P).query(H, workers=4)[0] <= 2 * voxel) if len(H) else 0.0

    rows = {}
    for name, P in [("DA3-alone", da3_pts), ("v3+bbox", comp_pts)]:
        f1, p1, r1 = fscore(P, gt_pts, 0.01)
        f5, p5, r5 = fscore(P, gt_pts, 0.05)
        rows[name] = dict(chamfer=chamfer_cm(P, gt_pts), f1=f1, f5=f5, iou=iou_vox(P, gt_pts, origin, voxel),
                          acc5=p5, comp5=r5, npts=len(P))
    rows["_holes"] = dict(shallow=rec(shallow, comp_pts), deep=rec(deep, comp_pts),
                          n_shallow=len(shallow), n_deep=len(deep))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", required=True)
    ap.add_argument("--cache-dir", default="outputs/tsdf_cache_replica_da3_006")
    ap.add_argument("--stitch-dir", default="outputs/fullroom_stitch_v3_da3")
    ap.add_argument("--csv", default="outputs/replica_v3_benchmark.csv")
    args = ap.parse_args()
    print(f"{'room':9s} {'method':11s} {'Chamfer↓':>9} {'F@1↑':>6} {'F@5↑':>6} {'IoU↑':>6} {'acc@5':>6} {'comp@5':>6}   holes(sh/deep)")
    agg = {}; csv = ["room,method,chamfer_cm,F@1,F@5,IoU,acc@5,comp@5,shallow_recall,deep_recall,npts"]
    for room in args.rooms:
        try:
            r = eval_room(room, args.cache_dir, args.stitch_dir)
        except Exception as e:
            print(f"{room:9s} FAILED: {e}"); continue
        h = r["_holes"]
        for m in ["DA3-alone", "v3+bbox"]:
            d = r[m]; hole = f"  {h['shallow']:.0f}%/{h['deep']:.0f}%" if m == "v3+bbox" else ""
            print(f"{room:9s} {m:11s} {d['chamfer']:8.2f}cm {d['f1']:5.1f} {d['f5']:5.1f} {d['iou']:5.1f} {d['acc5']:5.1f}% {d['comp5']:5.1f}%{hole}")
            agg.setdefault(m, []).append(d)
            sh = h['shallow'] if m == "v3+bbox" else ''; dp = h['deep'] if m == "v3+bbox" else ''
            csv.append(f"{room},{m},{d['chamfer']:.2f},{d['f1']:.1f},{d['f5']:.1f},{d['iou']:.1f},{d['acc5']:.1f},{d['comp5']:.1f},{sh},{dp},{d['npts']}")
    print("-" * 90)
    for m in ["DA3-alone", "v3+bbox"]:
        if m in agg:
            a = agg[m]; mean = lambda k: np.mean([x[k] for x in a])
            print(f"{'MEAN':9s} {m:11s} {mean('chamfer'):8.2f}cm {mean('f1'):5.1f} {mean('f5'):5.1f} {mean('iou'):5.1f} {mean('acc5'):5.1f}% {mean('comp5'):5.1f}%")
            csv.append(f"MEAN,{m},{mean('chamfer'):.2f},{mean('f1'):.1f},{mean('f5'):.1f},{mean('iou'):.1f},{mean('acc5'):.1f},{mean('comp5'):.1f},,,")
    with open(args.csv, "w") as f:
        f.write("\n".join(csv) + "\n")
    print(f"[saved] {args.csv}")


if __name__ == "__main__":
    main()
