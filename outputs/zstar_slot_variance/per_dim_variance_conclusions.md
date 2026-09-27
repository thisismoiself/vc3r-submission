# Conclusions: Per-Dimension Variance for `zstar` Targets

## Core Finding

The NOVA3R `zstar` target tokens should not necessarily be treated as fully fixed values.

When the same window is encoded multiple times from different point-cloud samples, the resulting token sets vary. After removing permutation ambiguity with Hungarian matching, each target slot has an empirical distribution around a mean.

For each slot `j`:

```text
mean[j]                 # (128,)
aligned_samples[:, j]   # (N, 128)
variance[j]             # (128,)
```

So each target token can be described as:

```text
zstar[j] ~ distribution(mean[j], variance[j])
```

rather than only:

```text
zstar[j] = fixed_vector[j]
```

## Slot-Dependent Variance

The variance is not uniform across the 768 token slots.

Some token slots are stable across point samples, while others vary much more. In the `room0`, start `0`, stride `20` experiment with 20 runs, final per-slot centered MSE showed a wide spread:

```text
q05  = 0.01598
q50  = 0.03566
q95  = 0.08012
q99  = 0.11935
max  = 0.17210
```

A high-variance slot around q95 has roughly 5x the variance of a low-variance q05 slot.

This suggests that the largest source of target uncertainty is dependent on which token slot is being analyzed.

## Per-Dimension Variance

Each slot is 128-dimensional, and the variance can be tracked per dimension:

```text
variance[j, d]
```

This allows the target to express that some dimensions of a token are more reliable than others.

A natural training target becomes:

```text
mu[j, d]      = mean value
sigma2[j, d]  = empirical variance
```

## Training Signal

With per-dimension variance, the loss can be weighted inversely to variance:

```text
loss[j, d] = (pred[j, d] - mu[j, d])^2 / sigma2[j, d]
```

This changes the training signal:

- Low-variance dimensions produce stronger gradients.
- High-variance dimensions produce weaker gradients.
- The adapter is encouraged to match stable parts of the target more tightly.
- Noisy or unstable target dimensions are not over-penalized.

The gradient becomes:

```text
d loss / d pred[j, d]
  = 2 * (pred[j, d] - mu[j, d]) / sigma2[j, d]
```

## Variance Noise Floor

Raw inverse variance can be unstable because very small measured variance can create excessively large weights.

Use a variance floor:

```python
var_floor = torch.quantile(var, 0.05)
var_eff = torch.clamp(var, min=var_floor)
```

Then train with:

```python
loss = ((pred - mu).square() / var_eff).mean()
```

This prevents dimensions with artificially tiny variance from dominating training.

## Recommended Cache Format

The k-means consensus cache should remain separate.

For the online-Hungarian distribution approach, cache:

```text
z_star_online_mean.pt       # (1, 768, 128)
z_star_online_var.pt        # (1, 768, 128)
z_star_online_var_eff.pt    # (1, 768, 128)
z_star_online_samples.pt    # (N, 768, 128)
```

Where:

```text
z_star_online_samples[:, j, :]
```

is the empirical distribution of matched tokens that produced:

```text
z_star_online_mean[:, j, :]
```

## Main Conclusion

The target should be modeled as a set of slot-wise distributions, not just a set of fixed token vectors.

The likely next training objective is:

1. Hungarian-match predictions to `z_star_online_mean`.
2. Use the same assignment to select `mean` and `var_eff`.
3. Train with variance-weighted MSE.

This preserves permutation invariance while making the loss aware of target uncertainty.
