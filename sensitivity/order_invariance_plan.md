# Order Invariance Plan

## Goal

Empirically verify whether the NOVA3R decoder is truly permutation-invariant
with respect to z_star token ordering. Everything built so far assumes this —
the k-means consensus, the Hungarian alignment, the Chamfer set loss. This test
confirms or refutes the assumption directly.

---

## Step 1 — Baseline decode

Load the cached val window (`start_0056`):
- `z_consensus`: shape `(1, 768, 128)`
- `pts_norm`: shape `(1, 8192, 3)`

Decode with a fixed seed → `pts_ref: (8192, 3)`.

---

## Step 2 — Decode stochasticity floor

Decode the *same* unshuffled `z_consensus` a second time with a different
seed → `pts_ref2`. Compute Chamfer distance between `pts_ref` and `pts_ref2`.

This is the baseline noise from the flow-matching ODE solver sampling
different initial noise. All shuffle Chamfer distances should be compared
against this floor — if shuffling produces distances at or below this value,
the decoder is order-agnostic.

---

## Step 3 — Shuffled decodes

For K=10 independent random permutations of the 768 token slots:

1. Sample a random permutation `π` of `[0, ..., 767]`.
2. Apply: `z_shuffled = z_consensus[:, π, :]` — shape `(1, 768, 128)`.
3. Decode with the *same fixed seed as step 1* → `pts_shuffled: (8192, 3)`.
4. Compute Chamfer distance between `pts_shuffled` and `pts_ref`.

Output: 10 Chamfer distances.

---

## Step 4 — Report

| Trial | Permutation seed | Chamfer vs ref |
|-------|-----------------|----------------|
| baseline (decode noise) | — | … |
| shuffle 0 | 0 | … |
| … | … | … |
| shuffle 9 | 9 | … |

Key question: are shuffle Chamfer distances at or near the decode stochasticity
floor (step 2), or significantly above it?

- **Near floor** → decoder is order-agnostic; ordering assumptions in the
  training pipeline are valid.
- **Significantly above floor** → decoder is order-sensitive; Hungarian
  alignment and Chamfer set loss are operating on a false premise.

---

## Step 5 — Decision

| Finding | Implication |
|---------|-------------|
| Shuffle distances ≈ decode noise floor | Order-agnosticism confirmed; proceed with set loss training |
| Shuffle distances >> decode noise floor | Decoder relies on token ordering; per-slot MSE with consistent ordering is the correct loss; Chamfer/Hungarian set loss is wrong |
