# Generalisation in the DA3→NOVA3R Adapter

Experiments: Q-Former adapter overfitting on stride-150 windows of Replica room0.

## The failure

Single-window overfit: MSE converges to ~1e-3 in 1000 steps — the adapter memorises one (DA3, z_star) pair.

10-window overfit with held-out validation (start=0, frames 0–1050):

| | Train MSE | Val MSE |
|---|---|---|
| Final (step 2000) | **0.003** | **3.95** |

Val loss diverges upward throughout training. The gap is 1000×. The adapter collapses to outputting the nearest memorised z_star rather than generalising.

## Root cause 1: z_star is stochastic

Script: `experiments/overfit_8frames/inspect_zstar_sampling_variance.py`

We encoded the same visible geometry (start=0, stride=20 window) ten times with different random subsampling seeds, then encoded a strictly disjoint held-out slice from the same visible pool.

| Test | MSE |
|------|-----|
| Pairwise MSE between seeds (same geometry) | **1.987** |
| Disjoint val slice vs each training seed | **1.997** |
| Disjoint val slice vs training mean z_star | 1.103 |

Two encodings of **identical geometry** differ by ~2.0 MSE. The NOVA3R encoder has no pressure to be consistent across random subsamples — it is trained with reconstruction loss only, and encoder/decoder co-adapt freely to whatever encoding happens to be convenient. Compare to a VAE, where KL divergence to a prior forces the encoder into a structured, regular space.

**Consequence:** a deterministic adapter with MSE loss cannot beat a ~2.0 floor regardless of architecture, training data, or coordinate frame, because the target itself is stochastic.

## Root cause 2: z_star is coordinate-frame-relative

`pts_norm` — the input to the NOVA3R encoder — expresses geometry relative to the first camera of each window. Windows with different starting frames produce z_stars in different coordinate frames. Even identical physical geometry encodes to numerically distant z_stars when expressed from different viewpoints.

To confirm this is not just theoretical, we computed the MSE between the val z_star (start=0, camera at frame 0) and each of the 10 training z_stars, alongside the camera rotation angle between each training window's first camera and the val camera:

| Sample | Start | z_star MSE to val | Camera rotation |
|--------|-------|-------------------|-----------------|
| 000    | 86    | 4.34              | 30.1°           |
| 001    | 173   | 4.48              | 16.1°           |
| 002    | 259   | 4.86              | 32.1°           |
| 003    | 345   | 4.60              | 20.8°           |
| 004    | 431   | 4.38              | 27.9°           |
| 005    | 518   | 4.45              | 41.9°           |
| 006    | 604   | 4.43              | 84.9°           |
| 007    | 690   | 4.31              | 112.6°          |
| 008    | 776   | 3.98              | 147.7°          |
| 009    | 863   | **3.88**          | **177.1°**      |

The nearest training z_star belongs to sample_009 — the window whose first camera is rotated **177°** from the val camera, nearly a perfect flip. The adapter's final val MSE (3.95) matches this almost exactly, confirming it collapses to outputting sample_009's z_star. Z_star distance and camera pose similarity are uncorrelated.

## k-means consensus z_star

Script: `experiments/overfit_8frames/kmeans_zstar_decode.py`

If root cause 1 is subsampling noise, the noise should be reducible by averaging. We tested this by:

1. Encoding the same geometry 10 times (different seeds) → 10 z_stars of shape (768, 128)
2. Stacking all 7680 tokens and running k-means with K=768 → 768 centroids of shape (128,)
3. Decoding the consensus z_star (1, 768, 128) via the NOVA3R flow-matching decoder

The resulting reconstruction is coherent and visually good. Point cloud centroids:

| | Centroid |
|---|---|
| k-means consensus | (0.023, 0.681, −0.284) |
| Single seed (seed-0) | (−0.059, 0.681, −0.212) |
| Disjoint val slice | (−0.057, 0.659, −0.214) |

Two findings:

**The decoder is order-agnostic.** K-means assigns tokens to clusters without preserving slot order, yet the decoder handles the result correctly. This means the flow-matching cross-attention treats z_star as a set, not a sequence.

**Subsampling noise is reducible.** The k-means consensus z_star decodes into a good reconstruction, confirming that the ~2.0 MSE noise across seeds is variance around a stable mean, not irreducible corruption. This directly suggests using consensus z_stars as training targets.

## Consecutive-window interpolation experiments

Scripts: `scripts/cache_consecutive_windows.py`, `scripts/train_consecutive_windows.py`, `scripts/interpolate_tokens.py`

To minimise the effect of coordinate frame change, we switched to stride-1 windows of 8 consecutive frames (camera spread 0.056–0.093m per window). Val window: frames 24–31 (start=24). Training windows: starts [0, 8, 16] on the left and [32, 40, 48] on the right — no frame overlap.

**Adapter training results — consensus targets, 2000 steps:**

| Config | Train MSE (final) | Val MSE (best) | Val MSE (final) | Diverges? |
|--------|-------------------|----------------|-----------------|-----------|
| 6 training windows (L3 R3) | 0.013 | **1.09** (step 200) | 1.67 | Yes |
| 14 training windows (L7 R7) | 0.007 | **1.04** (step 200) | 1.79 | Yes |
| 14 windows, Hungarian-aligned targets | 0.003 | **0.121** (step 600) | 0.138 | No |

Val loss peaks at step 200 in both raw-MSE runs then diverges steadily. Doubling the training data from 6 to 14 windows improved the best val loss by less than 0.05, while the final val loss worsened — more data did not slow divergence.

**Hungarian pre-alignment resolves the divergence.** Script: `scripts/train_hungarian.py`. Before training, every window's consensus z_star is reordered via Hungarian matching to align its token slots to the first training window's ordering (start=0 as reference). The cost matrix is (768×768) pairwise L2 distances; `scipy.optimize.linear_sum_assignment` finds the optimal bijection. MSE to the reference drops from ~1.8–2.1 (arbitrary k-means ordering) to 0.03–0.52 across training windows, and 0.26 for the val window — confirming the ordering was genuinely scrambled.

With aligned targets, val loss drops from 1.72 at step 1 to 0.12 by step 600 and stays there. No divergence. Final val MSE 0.138 — a 13× improvement over the raw-MSE run and well below the 1.13 weighted-interpolation baseline. The adapter is now learning to interpolate rather than memorise.

The root cause of the previous divergence was the arbitrary slot ordering of k-means centroids across windows. Per-slot MSE forced the adapter's Q-Former queries to reconcile contradictory targets: slot 47 in window A and slot 47 in window B encoded unrelated features. Hungarian alignment to a single reference gives each query a semantically consistent target across all training windows.

**Direct z_star interpolation — k-means consensus targets (no adapter):**

To test whether z_star space is locally smooth independently of the adapter, we interpolated the training consensus z_stars (k-means) directly at the val position and decoded the result.

| Method | MSE vs GT |
|--------|-----------|
| Weighted 1/dist (all 6 windows) | **1.13** |
| Linear (nearest neighbours: 16+32)/2 | 1.38 |
| Cubic spline (all 6 windows) | 1.96 |
| Quadratic (3 nearest: 8, 16, 32) | 2.00 |

Key findings:

**z_star space is locally smooth for simple averaging.** Weighted interpolation (MSE 1.13) outperforms the adapter (MSE 1.67) without any training. This confirms the problem is a learning failure, not a fundamental smoothness failure.

**Higher-order interpolation is worse.** Cubic spline and quadratic overshoot — z_star values do not vary smoothly enough across windows for polynomials to generalise correctly into the gap (Runge's phenomenon). Simple weighted averaging is the best interpolation strategy.

**The adapter cannot learn what interpolation achieves for free.** The adapter trained on the same 6 windows with 2000 gradient steps gets MSE 1.67; a closed-form weighted average gets 1.13. Scaling to 14 windows does not close the gap — best val MSE is 1.04 at step 200, still worse than weighted interpolation. This means the adapter is not learning the correct interpolation function from DA3 tokens. The failure is in the objective and architecture, not the amount of training data.

**Direct z_star interpolation — per-slot mean targets (failed):**

Switching the cache to per-slot mean dropped interpolation MSEs to 0.053–0.097, but both GT and interpolated reconstructions collapsed visually. Rescaling the mean back to the original l2 norm (~585) did not fix the collapse — the problem is not the magnitude but the fact that slots are not semantically consistent across seeds. Averaging slot i across seeds averages unrelated features, producing nonsensical tokens regardless of norm. The cache was reverted to k-means.

## Diagnosis

Three sources of variance, ranked by severity:

1. **Stochastic encoding** (~2.0 MSE floor, confirmed, reducible) — z_star is not a deterministic function of the geometry. Random subsampling of 8192 points introduces noise, but k-means averaging across seeds recovers a stable, decodable target.
2. **Coordinate frame** (additional ×2 gap, confirmed) — z_star is per-window-origin-relative; different windows describe the same room from different frames.
3. **Visibility** (secondary) — different windows see different geometry through their frustums.

## What needs to change

**Option A — clustered z_star targets.**

For each training window, encode the geometry N times and cluster the N×768 tokens into 768 centroids via k-means. Use this consensus z_star as the adapter's regression target. This eliminates root cause 1 without changing the decode pipeline. The coordinate frame problem (root cause 2) remains: the expected val MSE floor would drop from ~4.0 to ~2.0 (the frame gap alone), but cross-window generalisation would still require canonical pose normalisation.

An alternative to k-means is the **per-slot mean**: average slot *i* across the N seeds directly (shape (N, 768, 128) → mean over N → (768, 128)). This preserves slot ordering, which the decoder may rely on, and is the minimum-variance estimator per slot. It is appropriate when the encoder assigns similar geometry to the same slot consistently across seeds — likely for near-identical inputs. K-means is preferable if slot identity is not consistent across seeds.

**Per-slot mean collapses — even with norm rescaling.** When tested, the per-slot mean produced degenerate reconstructions regardless of whether the norm was corrected. The root cause is not the magnitude but the slot semantics: z_star has no pressure for slot consistency across seeds, so slot i in seed 0 and slot i in seed 7 encode unrelated features. Averaging them produces a nonsensical token at each position. Rescaling back to l2 ≈ 585 recovers the right magnitude but not meaningful content. K-means avoids this by finding the 768 most representative actual tokens from the full distribution — real tokens the decoder has seen, with no averaging across unrelated slots. The decoder's order-agnosticism is what makes k-means viable.

**Option B — replace z_star with `pts_norm` as the regression target.**

`pts_norm` is pure geometry — (N, 3) xyz coordinates with natural Euclidean structure. Deterministic for a fixed subsample, no noise floor, and a natural output for a depth-based backbone. The adapter predicts a point cloud; that point cloud is fed to the NOVA3R encoder at inference time to obtain z_star for decoding. Requires canonical pose normalisation to address root cause 2.

Both options require adding canonical pose normalisation to fully address cross-scene generalisation. Option A keeps z_star in the training loop and is a smaller change; Option B sidesteps the latent space entirely.

**What does not need to change:** adapter architecture (Q-Former), NOVA3R encoder/decoder weights, decode pipeline.

**Confirmed fix — Hungarian target alignment.** The dominant cause of training divergence was not the architecture or data quantity, but the arbitrary slot ordering of k-means consensus targets across windows. Hungarian pre-alignment to a single reference window resolves divergence entirely (val MSE 0.138 vs 1.79 without it). This fix is now part of the training pipeline (`scripts/train_hungarian.py`). The remaining gap to a perfect prediction is due to the coordinate frame problem (root cause 2), which is being addressed separately.

## Online Hungarian matching — 50-window room0 run

Script: `scripts/train_hungarian.py` (updated to online matching).

The pre-alignment approach requires a reference window; slot semantics are tied to start=0's ordering. The alternative is to compute the optimal bijection between predicted and GT tokens at every training step (`hungarian_mse()`), removing the reference-window dependency. The assignment is recomputed as the adapter improves and is detached from the gradient.

**Setup:** 50 training windows evenly spaced at stride 40 across the full 2000-frame room0 trajectory (starts 0, 40, 80, …, 1960). Five validation windows spaced ~400 frames apart (starts 20, 420, 820, 1220, 1620) — no frame overlap with training. Val loss is the mean Hungarian MSE across all 5 val windows.

**Training results:**

| Step | Train MSE | Val MSE |
|------|-----------|---------|
| 1 | 3.918 | 1.711 |
| 200 | 0.117 | 0.112 ← best so far |
| 400 | 0.070 | 0.079 |
| 800 | 0.043 | 0.065 |
| 1000 | 0.035 | 0.061 |
| 1600 | 0.017 | **0.060 ← best** |
| 2000 | 0.009 | 0.061 |

Best val MSE **0.060** at step 1600. No divergence. Checkpoint: `outputs/consecutive_windows/hungarian_online_v20_420_820_1220_1620_best.pt`.

**Decoded point cloud quality — Chamfer distance GT↔Pred (world space):**

| Val start | Chamfer | Centroid err |
|-----------|---------|--------------|
| 20 | 0.064m | 0.088m |
| 420 | 0.048m | 0.040m |
| 820 | 0.090m | 0.061m |
| 1220 | 0.083m | 0.137m |
| 1620 | 0.089m | 0.044m |
| **mean** | **0.075m** | **0.074m** |

The decode stochasticity floor (same z_star, different ODE seed) is 0.026m, so the adapter adds ~0.05m of error on top of irreducible decoder noise. Mean centroid error of 0.074m against a room ~5–6m across is <2% of scene extent. No outlier points in any val window.

**Cross-scene zero-shot — room1:**

The same checkpoint was evaluated on a single window from room1 (start=900, frames 900–907), a scene never seen during training.

| | Chamfer | Centroid err |
|--|---------|--------------|
| Room0 val mean | 0.075m | 0.074m |
| Room1 zero-shot | **0.138m** | 0.132m |

Chamfer is 1.85× worse on room1, but the adapter does not fail catastrophically — the predicted scene is in roughly the right location. The gap is consistent with the canonical pose problem: room1 has a different geometry and camera trajectory distribution, and the adapter has no pose-normalised representation to bridge scenes. Multi-scene training with room1 windows included would be the next step.

## Idea: Hungarian-aligned consensus at cache time

### Motivation

The current k-means consensus cache encodes geometry 20 times, stacks all 20×768 tokens, and clusters into 768 centroids. The resulting z_star is decodable but has two weaknesses:

1. **Arbitrary cluster ordering** — k-means assigns centroids in no particular order, so slot *i* has no consistent semantic meaning across windows. Online Hungarian matching in training compensates for this, but at the cost of 50–100 costly assignments per step (the current bottleneck: ~35s/step with 50 windows on CPU).

2. **Centroids are not averages** — k-means centroids are the means of each cluster, but each cluster may contain tokens from different seeds encoding different geometry. The result is a real-valued point in token space that the decoder has seen (since NOVA3R is order-agnostic), but it is not a principled average of the encoding noise.

The raw per-slot mean (average slot *i* across seeds) was previously shown to collapse, because slot *i* in seed 0 and slot *i* in seed 7 encode unrelated geometry — the encoder has no pressure for slot consistency across subsampling seeds.

### Proposed scheme

**Level 1 — better within-window consensus.**

Instead of k-means, align seeds 1–19 to seed-0 via Hungarian matching, then take the per-slot mean:

1. Encode geometry 20 times → z_stars {Z_0, Z_1, …, Z_19}, each (768, 128).
2. For each seed k ≥ 1, find the optimal bijection σ_k such that Z_k[σ_k(i)] ≈ Z_0[i] for all i (i.e., minimise Σ_i ‖Z_k[σ_k(i)] − Z_0[i]‖²).
3. Reorder each Z_k according to σ_k, then take the per-slot mean: Z_consensus[i] = mean_k Z_k^aligned[i].

After alignment, slot *i* across all 20 seeds refers to whichever token in each seed was closest to seed-0's slot *i* — genuinely similar features. The per-slot mean then averages related features and is the minimum-variance estimator per slot. This avoids the earlier per-slot-mean failure (which arose from averaging unrelated slots) while producing smoother, lower-variance targets than k-means centroids.

Cost: 19 × 768×768 Hungarian matchings per window ≈ 5–10s extra per window. Negligible relative to 20 NOVA3R encodes.

**Level 2 — eliminate Hungarian from training entirely.**

If each window's consensus is further aligned to a single global reference z_star (e.g. a medoid window chosen from the full cache), all cached targets share the same slot ordering. Training then uses simple per-slot MSE — no Hungarian matching per step. Expected speedup: ~50–100× per step (from ~35s to <1s with 50 windows).

This is the pre-alignment approach from the consecutive-window experiments (val MSE 0.138, no divergence), but done at cache time with a better consensus target and a principled reference choice, rather than at training-setup time with raw k-means z_stars.

### Empirical test

Script: `experiments/test_hungarian_consensus.py`

Tested on 5 windows from room0 (starts 0, 400, 800, 1200, 1600). Both methods encode 20 seeds; each consensus z_star is decoded and compared to the visible geometry via Chamfer distance. The noise floor is the Chamfer between two decodes of the same k-means z_star with different ODE seeds.

| Start | K-means | Hungarian mean | Noise floor | Winner |
|-------|---------|---------------|-------------|--------|
| 0 | 0.1035m | 0.1083m | 0.0324m | kmeans |
| 400 | 0.0774m | 0.0771m | 0.0328m | hungarian |
| 800 | 0.0866m | 0.0876m | 0.0468m | kmeans |
| 1200 | 0.0840m | 0.0876m | 0.0342m | kmeans |
| 1600 | 0.0723m | 0.0721m | 0.0338m | hungarian |
| **mean** | **0.0848m** | **0.0866m** | **0.0360m** | |

All differences are within the decode noise floor. K-means wins 3/5 windows; the means are essentially tied. Hungarian-aligned mean produces no better decoded output than k-means.

**Conclusion:** the benefit of Hungarian consensus caching is purely in training (consistent slot ordering enabling fast per-slot MSE), not in z_star quality. The k-means z_star is already a good enough regression target.

### Trade-off

The global reference is fixed. Online Hungarian adapts the assignment as the adapter improves; static pre-alignment commits to one ordering. For diverse multi-scene training, an outlier reference window could make the MSE landscape harder to optimise. Mitigation: choose the reference as the medoid of all cached z_stars (the window whose consensus is closest in mean L2 distance to all others).
