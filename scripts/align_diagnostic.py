#!/usr/bin/env python3
"""Per-window similarity ICP (R,t,scale) of pred -> input, to test whether the
stitch outliers are recoverable pose/scale errors or genuine shape errors.

Uses per_window.npz from stitch_office4.py (pred/oracle/input clouds in world frame).
Alignment target is each window's `input` (the real visible GT geometry of that window),
so this is a DIAGNOSTIC upper bound, not a deployable method.
"""
import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

def umeyama(X, Y):
    """Best s,R,t mapping X->Y (Kabsch + scale). X,Y: (N,3) corresponded."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = Xc.T @ Yc / len(X)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    var = (Xc ** 2).sum() / len(X)
    s = (S * np.array([1, 1, d])).sum() / var
    t = my - s * R @ mx
    return s, R, t

def icp_sim(src, dst, iters=40, trim=0.8):
    """Trimmed similarity ICP. Returns aligned src."""
    cur = src.copy()
    tree = cstree = cKDTree(dst)
    for _ in range(iters):
        d, idx = tree.query(cur, k=1)
        keep = d <= np.quantile(d, trim)           # reject worst (trim) correspondences
        s, R, t = umeyama(cur[keep], dst[idx[keep]])
        cur = (s * (R @ cur.T).T) + t
    return cur

def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (ta.query(b)[0].mean() + tb.query(a)[0].mean())

def main(npz_path):
    z = np.load(npz_path)
    pred, inp = z["pred"], z["input"]              # (W,Q,3), (W,Qi,3)
    W = pred.shape[0]
    print(f"{'win':>3} {'pred->in':>9} {'aligned':>9} {'gain%':>6}   {'s':>6} {'rot°':>6} {'t(m)':>6}")
    rows = []
    for i in range(W):
        p, q = pred[i].astype(np.float64), inp[i].astype(np.float64)
        before = chamfer(p, q)
        # recover transform for reporting
        cur = p.copy(); tree = cKDTree(q)
        s_acc = 1.0; R_acc = np.eye(3); t_acc = np.zeros(3)
        for _ in range(40):
            d, idx = tree.query(cur)
            keep = d <= np.quantile(d, 0.8)
            s, R, t = umeyama(cur[keep], q[idx[keep]])
            cur = (s * (R @ cur.T).T) + t
            s_acc *= s; R_acc = R @ R_acc; t_acc = s * (R @ t_acc) + t
        after = chamfer(cur, q)
        rot = np.degrees(np.arccos(np.clip((np.trace(R_acc) - 1) / 2, -1, 1)))
        gain = 100 * (before - after) / before
        rows.append((i, before, after, gain))
        print(f"{i:>3} {before*100:8.2f}c {after*100:8.2f}c {gain:5.1f}%   "
              f"{s_acc:6.3f} {rot:6.1f} {np.linalg.norm(t_acc):6.3f}")
    rows = np.array([(b, a) for _, b, a, _g in rows])
    print(f"\nmean pred->input   before={rows[:,0].mean()*100:.2f}cm  after={rows[:,1].mean()*100:.2f}cm")
    # outliers = before > 2x median
    med = np.median(rows[:,0])
    out = np.where(rows[:,0] > 2*med)[0]
    print(f"outlier windows (before>2x median {med*100:.2f}cm): {list(out)}")
    for i in out:
        print(f"  win {i}: before={rows[i,0]*100:.2f}cm  after={rows[i,1]*100:.2f}cm  "
              f"-> {'RECOVERED (pose/scale)' if rows[i,1] < 1.5*med else 'STILL HIGH (shape error)'}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "outputs/replica/stitch_office0_plain_midpoint/per_window.npz")
