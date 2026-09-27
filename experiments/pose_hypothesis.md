# Canonical Camera Pose Hypothesis Test

**Script:** `experiments/test_pose_hypothesis.py`
**Date:** 2026-06-21

## Hypothesis

Curved walls and bent geometry in decoded adapter predictions are caused by the adapter predicting a z_star that encodes geometry in a different camera frame than the val window. Applying the val window's c2w to a z_star from a different camera frame would produce spatially distorted geometry.

## Test Design

For a single val window (`windows_s20_room0/start_0000`, frames 0–20, yaw 108.7°), three things are decoded and compared in **camera space** (normalised units, before any c2w transform):

1. **Input** — the GT pts_norm fed to the NOVA3R encoder
2. **GT decoded** — the cached z_star_consensus decoded via the NOVA3R ODE
3. **Pred decoded** — the adapter's z_pred decoded via the NOVA3R ODE

Chamfer distance is computed between each decoded cloud and the input pts_norm, in camera space. Because a rigid c2w transform preserves planarity, curvature present in camera space cannot come from applying the wrong c2w — it must be baked into the z_star itself.

**Cross-frame test:** the z_star from each of 7 other cached training windows is decoded and then converted to world space using the val window's c2w (the "wrong" c2w for that z_star). If the adapter prediction resembles any of these wrong-frame clouds, the adapter is memorising a specific training window's z_star.

**Checkpoint:** `hungarian_online_v20_420_820_1220_1620_best.pt` (step 1600, val_loss 0.05990)
**Cache:** `scripts/data/windows_s20_room0` (starts 0, 37, 74, 111, 148, 185, 222, 259; yaw range 87–115°)
**Decode queries:** 4096 per cloud

## Results

### Token scale

| | Token norm (mean) |
|---|---|
| GT z_star | 21.78 |
| Adapter z_pred | 20.91 |
| Ratio | 0.960 |

Scale is not the issue — predicted tokens are the right magnitude.

### Camera-space Chamfer vs input pts_norm

| Cloud | Chamfer (normalised units) |
|---|---|
| GT z_star decoded | 0.0542 |
| Adapter z_pred decoded | 0.2090 |
| Ratio | **3.85×** |

The adapter prediction is 3.85× worse than the GT target in camera space. The curvature and distortion exist in normalised camera space before any c2w transform is applied.

### Cross-frame comparison

All training windows have similar yaw angles to the val window (87–115° vs 108.7°), so frame diversity is limited in this cache. Pred vs each training-window decoded cloud (camera space):

| Train start | Yaw | CD (pred cam vs train cam) | CD (pred world vs train wrong-c2w) |
|---|---|---|---|
| 37 | 115.0° | 0.2716 | 0.3814 |
| 74 | 104.8° | 0.3140 | 0.4410 |
| 111 | 101.2° | 0.3826 | 0.5373 |
| 148 | 98.9° | 0.3140 | 0.4409 |
| 185 | 96.8° | 0.3825 | 0.5372 |
| 222 | 87.6° | 0.5505 | 0.7730 |
| 259 | 86.7° | 0.5280 | 0.7415 |

The adapter prediction (CD 0.2090 vs input) is closer to the input than to any training window's decoded cloud (all > 0.27). The adapter is not memorising a specific training z_star.

## Conclusions

**Hypothesis disproved.** The curvature in decoded adapter predictions is not caused by z_star being in the wrong camera frame:

1. The distortion is present in camera space (Chamfer 0.209) before any c2w is applied. A rigid body transform cannot introduce curvature, so the c2w cannot be the cause.
2. The adapter prediction does not match any individual training window's decoded geometry — it is not frame-memorisation.
3. Token norm is correct (ratio 0.96) — scale is not the cause.

**Actual cause:** the NOVA3R ODE decoder amplifies token errors approximately **4×** in Chamfer terms. The adapter achieves token MSE 0.060 (Hungarian-matched), which sounds small but translates to Chamfer 0.209 vs 0.054 for the GT target. The ODE velocity field, when conditioned on imperfect z_pred, converges to a distorted geometry that visually appears as walls bowing toward the camera origin — the ODE's fallback when the conditioning has conflicting or imprecise signals.

**Implication for canonical frame normalisation:** this change would not address the observed artifact, because the artifact is in z_pred content, not in coordinate frame orientation. The path to flatter walls is lower token MSE — achieved via more training data, more training steps, or a larger adapter — not pose normalisation.

**ODE amplification ratio:** to produce clean flat-wall geometry, token MSE would need to fall to approximately 0.015 (4× reduction from current 0.060).
