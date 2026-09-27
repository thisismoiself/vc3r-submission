# DA3 → NOVA3R adapter: detail-recovery study (office4 held-out)

*Branch: `sensitivity-analysis`. All eval numbers below are read directly from the eval
logs in `experiments/overfit_8frames/eval_*_stitch.log` (not from memory). Held-out room
is **office4**; every model is decoded with the **midpoint** ODE solver at **50 000
points/window** and stitched with GT poses.*

---

## 1. Goal & the decisive diagnostic

The adapter (`DA3ToNOVA3RAlignment`) predicts NOVA3R's `768×128` latent tokens from DA3
image features; a **frozen** NOVA3R flow-matching decoder turns tokens → point cloud.
Goal: recover **fine detail** (chairs / table / corner objects) in the reconstruction
**without retraining the decoder or encoder**.

**Oracle = `decode(encode(real geometry))`** is the frozen-decoder ceiling. It reconstructs
crisp furniture ⇒ the detail *is present* in the 768×128 tokens, so the bottleneck is the
**adapter's token prediction**, not the decoder. We therefore measure everything against
the oracle, and add a **furniture-region F-score** as the detail metric (mean Chamfer is
dominated by bulk walls/floor and hides the furniture gap).

**Furniture F@2cm** — crop GT and pred to the interior height band
(`z ∈ [floor+0.15, floor+1.1]`, wall-margin 0.4 m) so only chairs/table/corner survive,
then F-score at a 2 cm threshold. `F = 2·P·R/(P+R)`, P = fraction of pred points within
2 cm of GT (accuracy), R = fraction of GT points within 2 cm of pred (completeness).
Region = **445 472 GT pts**. This is the number that actually tracks furniture sharpness.

---

## 2. Did the baseline change? (caching / training)

**No — the baseline is fully preserved.** The changes on this branch are *additive and
opt-in*; run with defaults and you reproduce the original var-weighted production recipe.

### Caching scripts — behavioural no-op
`cache_consecutive_windows.py`, `cache_online_hungarian_zstar_windows.py`: the **only**
change is redirecting the NOVA3R root from `nova3r/` to `nova3r_lib/` via a
`NOVA3R_DIR` env var (the bare `nova3r/` copy on this machine has no checkpoint). Same
`scene_ae` checkpoint, same encode, **same z_star target** (online-Hungarian consensus
mean + `var_eff`). The cached targets are numerically identical to before.

### Training script — additive flags, defaults = baseline
`train_online_var_hungarian.py`. With no new flags the run is the original recipe. New:

| addition | default | effect when default |
|---|---|---|
| `--loss-mode {var_weighted,plain,sqrt_var,huber}` | `var_weighted` | original loss |
| `--vel-loss-weight` (velocity aux loss) | `0.0` (off) | not used |
| `--max-per-root` / `--max-val-per-root` | `None` | no cap |
| `--lazy-da3` (LRU streaming loader for full-data) | off | preload as before |
| Hungarian solver: scipy → **lapjv** (`lap`) | — | **same optimal assignment**, ~20× faster |
| `GradScaler` API line (torch 2.2) | — | identical numerics |

The **only non-opt-in change** is the Hungarian matcher swap (scipy `linear_sum_assignment`
→ `lapjv`), which returns the *same optimal column assignment* — it changes speed
(~13 s/step → sub-second), not the loss value. **⇒ The baseline is byte-for-byte
reproducible**; every experiment below is a controlled variation from it.

---

## 3. Training procedure (the recipe)

- **Target:** per-window `z_star` = 30-seed online-Hungarian **consensus mean** of the
  encoder's tokens (+ per-token `var_eff`). Marginalises the point-sampling nuisance →
  Bayes-optimal target. *(Verified not over-smoothed: `decode(mean)` is as sharp as the
  median single encode.)*
- **Adapter:** DA3 features (layers [1,3], ≤2048 tokens, 392×518 imgs) → cross-attention
  → 768×128 tokens.
- **Optimiser:** AdamW, b24, lr 5e-4→1e-6 cosine w/ restart (t0 6000, mult 2), warmup 1000,
  EMA 0.999, weight-decay 1e-4, dropout 0.2, token-dropout 0.2, float16 cache.
- **Loss:** Hungarian-matched MSE (`plain` best — see §4) + optional velocity aux (λ=0.3).
- **Eval:** `stitch_office4.py --fm-sampling midpoint --num-queries 50000` → whole-scene
  Chamfer/F@5 + furniture-region F@2/F@5, vs oracle.

---

## 4. Results — all office4 held-out, midpoint decode, 50k pts/window

**Oracle ceiling (same across evals):** whole Chamfer **0.044**, whole F@1 **0.043**,
whole F@5 **0.812**; **FURN F@1 0.240**, FURN F@2 **0.551**, FURN F@5 **0.807**.
(Oracle = `decode(z*)`, the frozen-decoder ceiling.) Note the oracle's **whole F@1 (0.043) is
below the trained models' ~0.059** — confirming whole F@1cm is near-floor and *not* a quality
signal even for the ceiling; read **FURN F@1** for the tight-threshold comparison. Relative to
oracle, the best model reaches **58% of oracle FURN F@1 (0.140/0.240)** vs 67% at F@2 — the tighter
threshold shows the model is *relatively worse at the finest detail scale*.

### 4a. Loss ablation (fixed data, one variable)
| loss | whole Chamfer ↓ | whole F@1 ↑ | whole F@5 ↑ | **FURN F@1 ↑** | **FURN F@2 ↑** | FURN F@5 ↑ |
|---|---|---|---|---|---|---|
| var_weighted (baseline / production) | 0.0640 | 0.0583 | 0.661 | 0.107 | 0.310 | 0.597 |
| **plain** (unweighted) — **best** | 0.0649 | 0.0557 | 0.666 | 0.111 | **0.324** | 0.617 |
| huber (δ=1, robust) | 0.0695 | 0.0519 | 0.617 | 0.099 | 0.284 | 0.552 |

→ **plain wins.** Variance-weighting *down-weights exactly the high-variance detail tokens*
(furniture); huber smooths them away.

> **On the F@1cm columns (added per tutor request).** The tighter 1 cm threshold is more
> discriminative *on the furniture region* — **FURN F@1cm amplifies the gaps** between methods
> (e.g. full-recipe is **+26%** over plain at F@1 vs only +14.5% at F@2; vel0.3 is +8% at F@1 vs
> +4% at F@2). **Whole-scene F@1cm, however, is near-floor (~0.05) and NOT a reliable signal** —
> at 1 cm the full room is dominated by hard-to-hit wall/floor points, so it's noisy and can even
> *anti-rank* (vel0.3 is the best model yet has the lowest whole F@1). Read **FURN F@1** for detail;
> ignore whole F@1 except as the "raw" number. FURN F@1cm is shown in every table below.

### 4b. Frame-count / span ablation (plain loss, 70 windows/room, one variable)

**A "window" has two independent knobs**, both drawn as *ranges* per window at cache time:
- **span** = temporal length of the window (first→last frame, in raw-frame units) → sets the
  physical area it covers / token density (768 tokens ÷ extent).
- **n_frames** = number of images actually fed to the model within that span → sets
  completeness + supervision richness. Emergent **stride ≈ span/(n_frames−1)**.
- Deployment (office4 eval) is fixed at **stride 10, 8 frames, span 70** for *every* model.

**Frame-count axis** — vary #images, span fixed 24–100:
| distribution (cache) | n_frames | span | whole Chamfer ↓ | whole F@1 ↑ | whole F@5 ↑ | **FURN F@1 ↑** | **FURN F@2 ↑** | FURN F@5 ↑ |
|---|---|---|---|---|---|---|---|---|
| nf4–10 (`run1_span24_100_nf4_10`) | 4–10 | 24–100 | 0.0706 | 0.0455 | 0.616 | 0.096 | 0.267 | 0.531 |
| nf8 (`fc_nf8_span24_100`) | 8 | 24–100 | 0.0669 | 0.0521 | 0.636 | 0.109 | 0.302 | 0.568 |
| **nf16** (`fc_nf16_span24_100`) — **best** | **12–24** | 24–100 | 0.0660 | 0.0563 | 0.677 | 0.114 | **0.324** | 0.617 |

→ **More frames per window help monotonically** (nf16 > nf8 > nf4–10), even though deployment
is only 8 frames — richer multi-view supervision at train time generalises best.

**Span axis** — vary window extent, n_frames fixed at 8:
| distribution (cache) | n_frames | span | whole Chamfer ↓ | whole F@1 ↑ | whole F@5 ↑ | **FURN F@1 ↑** | **FURN F@2 ↑** | FURN F@5 ↑ |
|---|---|---|---|---|---|---|---|---|
| spanshort (`fc_spanshort_16_36_nf8`) | 8 | 16–36 | 0.0707 | 0.0391 | 0.572 | 0.099 | 0.278 | 0.564 |
| nf8 anchor (`fc_nf8_span24_100`) | 8 | 24–100 | 0.0669 | 0.0521 | 0.636 | 0.109 | 0.302 | 0.568 |
| spanlong (`fc_spanlong_120_300_nf8`) | 8 | 120–300 | 0.0639 | 0.0508 | 0.654 | 0.108 | 0.312 | 0.592 |

→ **Span is a mild, monotone-increasing lever, and shorter is worse:**
spanshort 0.278 < anchor 0.302 < spanlong 0.312 (FURN F@2). This **refutes the "concentrate
the 768 tokens on a small area → finer detail" hypothesis** — the opposite is true, longer
spans are marginally better (more scene context per window; short spans also mismatch the
deploy span of 70). But the whole span axis (0.278→0.312) is **weaker than the frame-count
axis** (0.267→0.324): **#frames, not span, is the dominant lever**, which is why the recipe
fixes nf16 and keeps a broad span (24–100).

### 4c. Velocity aux loss (plain + nf16, sweep λ)
| model | whole Chamfer ↓ | whole F@1 ↑ | whole F@5 ↑ | **FURN F@1 ↑** | **FURN F@2 ↑** | FURN F@5 ↑ |
|---|---|---|---|---|---|---|
| plain (λ=0) | 0.0649 | 0.0557 | 0.666 | 0.111 | 0.324 | 0.617 |
| **plain + velocity λ=0.3** — **best trained model** | **0.0632** | 0.0495 | 0.657 | **0.120** | **0.337** | **0.632** |
| plain + velocity λ=1.0 (too strong) | 0.0659 | 0.0549 | 0.649 | 0.117 | 0.307 | 0.581 |

→ Velocity-matching (match the frozen decoder's velocity field under `z_pred` vs target
tokens on NOVA3R's cosine FM path; permutation-invariant, no Hungarian) is the **first lever
to dent the residual *shape* error**: it cut the post-alignment shape residual ~11%
(4.99→4.43 cm). **λ=0.3 optimal; λ=1.0 overshoots** (worse than plain).

### 4d. DA3-depth RANSAC — GT-free outlier rejection (on the vel λ=0.3 model)
DA3's own predicted depth (unprojected with GT poses) is a **GT-free reference**:
DA3→GT = **2.7 cm**, and `corr(pred→GT error, pred→DA3 dist) = 0.913` — so
`pred→DA3` distance ranks bad windows without touching GT (refutes the circularity worry).

| selection | whole Chamfer ↓ | whole F@5 ↑ | **FURN F@1 ↑** | FURN F@2 ↑ |
|---|---|---|---|---|
| keep all 25 windows | 0.0632 | 0.657 | 0.120 | 0.337 |
| **drop-3 by DA3 (GT-free)** | **0.0540 (−15%)** | 0.686 | **0.125** | **0.349** |
| drop-3 by GT (ceiling of the method) | 0.0529 | 0.695 | 0.128 | 0.352 |

→ GT-free RANSAC recovers **~85–90% of the GT-selection ceiling**; drops the blown-up
windows (12/18/22, `pred→GT` 0.08–0.49 m). Cleaned PLY:
`outputs/replica/stitch_office4_vel_lambda03_midpoint/office4_pred_da3ransac_drop3.ply`.

### 4e. Full-recipe data-scaling run (nf16 **1225** windows + plain + velocity λ=0.3) — **best trained model**
Job `1612078`, 18k steps, best @ step 15200: **val_plain_mse 0.0508** (vs ~0.054 on the
≤700-window runs — a clear generalisation gain from ~1.75× more data).
Checkpoint `outputs/consecutive_windows/fullrec_nf16_vel03_best.pt`.

| model | whole Chamfer ↓ | whole F@1 ↑ | whole F@5 ↑ | **FURN F@1 ↑** | **FURN F@2 ↑** | FURN F@5 ↑ |
|---|---|---|---|---|---|---|
| vel λ=0.3 (≤700 windows) | 0.0632 | 0.0495 | 0.657 | 0.120 | 0.337 | 0.632 |
| **full-recipe (1225 windows)** — **best** | **0.0567** | 0.0586 | **0.707** | **0.140** | **0.371** | **0.667** |

→ **Data scaling is the single biggest trained-model gain:** FURN F@2 0.337→**0.371**,
whole F@5 0.657→**0.707**, whole Chamfer 0.063→**0.057** — all without RANSAC. It also
shrinks the outlier windows (win12 0.081→0.070; win22 0.181→0.176; only win18 still blows
up at 0.354) ⇒ DA3-RANSAC (§4d) should stack on top for a further whole-scene gain.

---

## 5. Story for the report (one line each)

1. **Oracle diagnostic** proves detail lives in the tokens → fix the *adapter*, not the decoder.
2. **Midpoint solver** ≫ Euler: ~75% of decode scatter was integration error — free (oracle whole F@5 0.46→0.81).
3. **Loss = plain** beats var-weighting/huber on furniture (0.324 vs 0.310 vs 0.284).
4. **More frames** help (nf16 0.324 > nf8 0.302 > nf4–10 0.267); **span is a weaker lever** and **shorter span is worse** (spanshort 0.278 < anchor 0.302 < spanlong 0.312) — the "small span → finer detail" hypothesis is ruled out.
5. **Velocity aux loss (λ=0.3)** fixes genuine *shape* residual → best trained model (FURN F@2 0.337).
6. **DA3-depth RANSAC** is a GT-free front-end that removes outlier windows → whole Chamfer −15%, FURN F@2 0.349.
7. **Data scaling** (1225 windows, full recipe) → **best trained model**: FURN F@2 **0.371**, whole F@5 **0.707**, Chamfer **0.057**.

**Progression (FURN F@2):** baseline 0.310 → plain 0.324 → +velocity 0.337 →
**+data-scaling 0.371** → (+DA3-RANSAC expected to stack further). Oracle 0.55.
**Whole-scene Chamfer:** baseline 0.064 → full-recipe **0.057** → RANSAC-on-vel03 0.054
(RANSAC on the full-recipe model not yet run). Oracle 0.044.

### Rule-outs (negative results — kept deliberately)
- Shorter span does not sharpen detail (token density is not the lever).
- The consensus-mean target is not over-smoothed (`decode(mean)` = median sample sharpness).
- The decoder is not the bottleneck (oracle is crisp).

---

## 6. Remaining gap & next steps
- **DA3-RANSAC on the full-recipe model** (immediate): §4d gave −15% whole Chamfer on the
  weaker vel λ=0.3 model; win18 (0.354) still blows up in the full-recipe run, so RANSAC
  should stack for a further whole-scene gain on top of FURN F@2 0.371.
- **DA3 alignment of kept windows** (designed, not run): the align diagnostic shows ~44%
  of the residual is recoverable per-window pose/scale; DA3 depth can fix it GT-free, stacking on RANSAC.
- **Decoder fine-tuning** is the *only* route to exceed the oracle — not needed unless the
  adapter provably saturates below it (it has not).

## 7. Key files
- Trainer: `scripts/train_online_var_hungarian.py` (loss modes, velocity loss, lazy loader)
- Caches: `scripts/cache_online_hungarian_zstar_windows.py`, `scripts/cache_consecutive_windows.py`
- Eval: `scripts/stitch_office4.py` (midpoint + furniture crop), `scripts/align_diagnostic.py`
- DA3 reference / RANSAC: `scripts/da3_ref_test.py`
- SLURM: `scripts/slurm/{cache_nf16_full,train_fullrecipe,train_loss_all,train_framespan_all}.sbatch`
