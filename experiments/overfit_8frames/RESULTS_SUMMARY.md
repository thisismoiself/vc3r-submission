# DA3→NOVA3R adapter — detail-recovery results (office4 held-out)

## Goal & framing
Adapter predicts NOVA3R latent tokens from DA3 image features; a **frozen** NOVA3R
flow-matching decoder turns tokens → point cloud. Goal: recover **fine detail**
(chairs / furniture) in the office4 reconstruction **without** retraining the decoder.

**Decisive diagnostic — the ORACLE.** `oracle = decode(encode(real geometry))` is the
frozen-decoder ceiling. It reconstructs crisp chairs ⇒ **the detail is present in the
768×128 tokens**; the bottleneck is the *adapter's token prediction*, not the decoder.
We therefore measure everything against the oracle on held-out **office4**, with a
**furniture-region F-score** (crop to chairs/table/corner) as the detail metric.

## The pred→oracle gap decomposes into three parts (each measured)
1. **Decode-sampler noise** — fixed (midpoint solver).
2. **Per-window pose/scale** — **~44% of the residual**, recoverable by a similarity transform.
3. **Genuine shape error** — the rest (post-alignment residual ~4.4 cm).

## Results (all office4, midpoint decode, 50k pts/window)
| model | whole Chamfer ↓ | whole F@5 ↑ | **FURN F@2 ↑** | FURN F@5 ↑ |
|---|---|---|---|---|
| var-weighted loss (production choice) | 0.064 | 0.661 | 0.310 | 0.597 |
| **plain** Hungarian MSE | 0.065 | 0.666 | 0.324 | 0.617 |
| **plain + velocity loss (λ=0.3) — BEST** | **0.063** | 0.657 | **0.337** | **0.632** |
| — frame count: nf4–10 / nf8 / **nf16** | 0.071/0.067/0.066 | | 0.267/0.302/**0.324** | |
| — span shorter (16–36) | 0.071 | 0.572 | 0.278 | 0.564 |
| **ORACLE (frozen-decoder ceiling)** | **0.044** | **0.813** | **0.553** | **0.812** |

## What we learned (controlled, one variable at a time)
- **Midpoint ODE solver** ≫ default Euler: ~75% of decode point-scatter was *integration
  error*, not the model. Oracle whole-scene Chamfer 0.069→0.044, F@5 0.46→0.81 — **free**.
- **Loss = plain** beats variance-weighting (which *down-weights the high-variance detail
  tokens* = chairs) and Huber (which smooths). Plain FURN F@2 0.324 vs 0.310 vs 0.284.
- **More frames per window help** (nf16 12–24 > nf8 > nf4–10). Richer multi-view
  supervision generalizes best even to the 8-frame deployment.
- **Velocity-matching loss** (geometry-aware: match the frozen decoder's velocity field
  under predicted vs target tokens, on NOVA3R's cosine FM path) **improves detail**
  (FURN F@2 0.324→0.337) and **reduces the post-alignment shape residual 11%**
  (4.99→4.43 cm) ⇒ it fixes genuine *shape*, not placement.

## Negative results (rule-outs — important for the report)
- **Shorter span does NOT help** — refutes the "concentrate tokens → finer detail"
  hypothesis (spanshort is the *worst* whole-scene). Train-test span mismatch + lost context.
- **The 30-seed consensus mean target is NOT over-smoothed** — `decode(mean)` is as sharp
  as the median single encode, and the mean is the Bayes-optimal target (the adapter
  cannot see the point-sampling nuisance). ⇒ keep the mean.
- **Decoder is not the bottleneck** — oracle is crisp; capacity is fine.

## Best run
**plain loss + velocity λ=0.3 + midpoint** — FURN F@2 **0.337**, whole Chamfer **0.063**.
Closes part of the gap toward oracle (0.553 / 0.044), with the velocity loss being the
first lever to dent the *residual shape* error specifically.

## Remaining gap & next steps (in progress / planned)
- **Data scaling (running):** the recipe is data-starved (≤700 windows). Caching the
  winning **nf16** distribution to ~1225 windows; train the full recipe to convergence.
  More data is the most reliable fix for the held-out generalization gap.
- **DA3-depth alignment (designed):** removes the ~44% pose/scale part GT-free by aligning
  each decoded window to DA3's own predicted depth. *Caveat:* NOVA3R is built from DA3
  features, so we must test for correlated/circular error before committing.
- **Decoder fine-tuning (eventual):** the oracle *is* the frozen-decoder ceiling; to go
  *beyond* it would require fine-tuning the decoder (the only route to exceed oracle).

## Infra notes
Shared GPU (serialize) + SLURM cluster (request fast GPUs, external QOS caps 2 jobs).
All evals: `stitch_office4.py --fm-sampling midpoint --num-queries 50000`; furniture crop
+ similarity-ICP alignment diagnostic in `align_diagnostic.py`.
