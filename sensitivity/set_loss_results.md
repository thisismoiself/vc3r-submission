# Set Loss Experiment Results

## Context

Testing different training objectives for the DA3→NOVA3R Q-Former adapter on
14 consecutive windows (val=56, L7R7, Replica room0). The original per-slot MSE
loss diverged because k-means consensus targets have arbitrary slot ordering
across windows — the adapter was forced to reconcile contradictory targets.

The order invariance test confirmed the NOVA3R decoder is fully order-agnostic
(shuffle/floor ratio 0.01×), making per-slot MSE structurally wrong and
set-valued losses the correct approach.

---

## Results

| Method | Val loss (best) | Val loss (final) | Pred centroid error | Diverges? |
|--------|----------------|-----------------|-------------------|-----------|
| Raw per-slot MSE | 1.04 (step 200) | 1.79 | large | Yes |
| Hungarian pre-align + MSE | **0.121** (step 600) | **0.138** | small | No |
| Chamfer set loss (no pre-align) | 2.16 (step 1000) | 2.26 | large | No (collapsed) |

---

## Chamfer set loss — detail

Script: `scripts/train_setloss.py`

Training with Chamfer distance between predicted and GT token sets, no
pre-alignment of targets. Loss scale is not directly comparable to MSE.

| Step | Train Chamfer | Val Chamfer |
|------|--------------|------------|
| 1 | 22.14 | 13.50 |
| 200 | 2.05 | 2.47 |
| 600 | 1.82 | 2.40 |
| 1000 | 1.67 | 2.16 |
| 1400 | 1.65 | 2.19 |
| 2000 | 1.63 | 2.26 |

GT centroid: `[-0.143, 0.252, -0.286]`
Pred centroid: `[-0.741, 0.708, -0.661]` — far from GT.

**Failure mode: many-to-one collapse.** Chamfer allows multiple predicted
tokens to map to the same GT token. The adapter collapsed onto a small set of
"average" tokens early in training (loss plateaus after step 200 and barely
moves for 1800 steps). The gradient signal is too weak and inconsistent to
drive further convergence. The adapter cannot even fit the training data
(train Chamfer 1.63 after 2000 steps).

---

## Hungarian pre-alignment — detail

Script: `scripts/train_hungarian.py`

All z_stars (train + val) are reordered offline via Hungarian matching to align
token slots to the first training window (start=0) as reference. Regular MSE
loss on aligned targets.

| Step | Train MSE | Val MSE |
|------|-----------|---------|
| 1 | 3.76 | 1.87 |
| 200 | 0.144 | 0.156 |
| 400 | 0.072 | 0.126 |
| 600 | 0.042 | 0.134 |
| 1000 | 0.018 | 0.138 |
| 2000 | 0.003 | 0.142 |

GT centroid: `[-0.143, 0.252, -0.286]`
Pred centroid: `[-0.117, 0.242, -0.323]` — close to GT.

No divergence. Val loss stable in 0.12–0.18 range throughout.

**Limitation:** requires a reference window. Adapter's learned slot semantics
are tied to start=0's ordering. New scenes or sequences need a new reference.

---

## Next experiment

Online Hungarian matching loss: compute the optimal bijection between predicted
and GT token sets at every training step (no pre-alignment, no reference
window). Enforces a bijection unlike Chamfer, removing the many-to-one
collapse. Removes the reference-window dependency of pre-alignment.

This combines the stability of Hungarian pre-alignment with the
reference-independence of the set loss approach.
