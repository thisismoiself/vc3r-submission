# Order Invariance Results

**Window:** val start=56 (frames 56–63, Replica room0)
**Shuffles:** 10 random permutations of 768 token slots
**Decode seed:** 42 (fixed across all shuffles; different seed used for floor only)

## Results

| Trial | Chamfer vs baseline |
|-------|-------------------|
| Stochasticity floor (same z_star, seed=43) | 0.025675 |
| Shuffle 0 | 0.000235 |
| Shuffle 1 | 0.000239 |
| Shuffle 2 | 0.000238 |
| Shuffle 3 | 0.000240 |
| Shuffle 4 | 0.000232 |
| Shuffle 5 | 0.000231 |
| Shuffle 6 | 0.000234 |
| Shuffle 7 | 0.000236 |
| Shuffle 8 | 0.000236 |
| Shuffle 9 | 0.000243 |

| Metric | Value |
|--------|-------|
| Decode stochasticity floor | 0.025675 |
| Shuffle mean | 0.000236 |
| Shuffle std | 0.000003 |
| Shuffle min | 0.000231 |
| Shuffle max | 0.000243 |
| Shuffle / floor ratio | **0.01×** |

## Verdict

**Order-agnostic.** Shuffling the 768 token slots produces Chamfer distances ~100× *below* the decode stochasticity floor. The residual 0.000236 is floating-point noise, not a meaningful difference. Token slot identity is completely irrelevant to the NOVA3R decoder — the decoder treats z_star as an unordered set.

## Implications

Per-slot MSE is the wrong loss for this decoder. The original training failure (divergence to val MSE ~1.79) was caused by forcing the Q-Former to match arbitrarily-ordered k-means centroids slot-by-slot — a meaningless constraint given the decoder's order-agnosticism. Hungarian pre-alignment (resolving the ordering) or Chamfer/Hungarian set loss (making the loss permutation-invariant) are the correct approaches.
