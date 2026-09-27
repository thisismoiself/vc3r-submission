# Adapter — Update Report (later experiments)

Follows on from `ADAPTER_DETAIL_REPORT.md`. Same protocol throughout: office4 held-out,
midpoint decode, 50k pts/window, 25 non-overlapping 8-frame windows stitched with GT poses,
evaluated vs the 2M-pt GT mesh with the furniture-region crop as the detail metric.

**Reference points (from the main report):**
- Best trained model = **full-recipe** (nf16, 1225 windows, plain + velocity λ0.3): **FURN F@2 0.371**, whole Chamfer **0.0567**.
- **Oracle ceiling** (`decode(z*)`): FURN F@1 **0.240**, FURN F@2 **0.551**, whole Chamfer **0.044**.

**One-line outcome of everything below:** loss-shape tricks (velocity, endpoint) give only small
or negative furniture gains; the big lever is **information** (clean geometry: +0.032). Two findings
recur and explain why: **(a) token-MSE ⟂ furniture detail**, and **(b) plain geometric losses are
bulk-dominated**.

---

## 1. DA3-RANSAC on the best model (GT-free outlier rejection)
The main report ran RANSAC on the vel-λ0.3 model; here on the **full-recipe** best model.
Rank the 25 windows by `pred→DA3` distance (GT-free; DA3→GT = 2.7 cm, corr with true error 0.90),
drop the worst 3, re-stitch.

| selection | whole Chamfer ↓ | whole F@5 ↑ | FURN F@2 ↑ | FURN F@5 ↑ |
|---|---|---|---|---|
| keep all 25 | 0.0567 | 0.707 | 0.371 | 0.667 |
| **drop-3 by DA3 (GT-free)** | **0.0490 (−14%)** | 0.737 | **0.386** | 0.680 |
| drop-3 by GT (upper bound) | 0.0489 | 0.734 | 0.339 | 0.648 |

→ Stacks on the best model (whole Chamfer −14%, FURN F@2 0.371→0.386). Filtered clouds saved as
`office4_pred_ransac_drop3_da3.ply` (GT-free) / `_gt.ply`. Note the GT-free selection *beats* the
GT-error selection on furniture (0.386 vs 0.339): ranking by whole-scene error drops a
good-furniture/bad-wall window, so `pred→DA3` is the better furniture selector here.

---

## 2. Token-noise sensitivity analysis — *why loss magnitude is not the lever*
Perturb the true tokens `z*` with isotropic Gaussian noise of growing scale, decode with a fixed
query init (isolates token effect), measure how far the decoded cloud drifts from the clean decode.
(`scripts/token_noise_sensitivity.py`, 5 held-out windows.)

- **Sampler floor** (decode `z*` twice, different query seeds): **2.5 cm** — irreducible.
- **Adapter's operating point:** its token error (val_mse 0.051) corresponds to isotropic noise
  that moves the cloud only **~2.5 cm** — i.e. **at the sampler floor** (decoder fully absorbs it).
- **But the adapter's *actual* decode deviation from oracle** (`chamfer(decode(z_pred), decode(z*))`)
  = **6.97 cm mean / 5.34 cm excl-3-worst** — about **2–2.7× worse** than isotropic noise of the
  same magnitude; equivalently, it does the damage of ~6× the MSE.

**Conclusion: the adapter's token error is STRUCTURED, not just large** — concentrated in the
decoder-sensitive / detail directions. Lowering *average* token-MSE (e.g. more data) is inefficient;
the levers must target the error's **structure** (decoded-space losses, detail weighting, geometry).
This is the "token-MSE ⟂ detail" theme, confirmed in point space.

---

## 3. Geometry conditioning — inject 3D into the adapter
**Mechanism:** add a point cloud as an extra cross-attention *source* stream — Fourier-embed the
points → MLP → hidden dim, tag with a learned modality embedding, concatenate to the DA3 image
tokens so the target queries attend over metric geometry, not just appearance. Backward-compatible
(off ⇒ identical to baseline). `--geom-cond` in the trainer; `stitch_office4.py --geom-source`.

### 3a. Step 1 — clean geometry (`pts_norm`, upper bound) — **WORKS**
Feed the (GT-derived) visible cloud the pipeline already has. Same recipe + `--geom-cond`.

| model | whole Chamfer | whole F@5 | **FURN F@2** | FURN F@5 | val_mse |
|---|---|---|---|---|---|
| baseline (appearance-only) | 0.0567 | 0.707 | 0.371 | 0.667 | 0.0508 |
| **+ geometry (pts_norm)** | 0.0567 | 0.715 | **0.403 (+8.7%)** | 0.689 | 0.0524 |

→ **+0.032 FURN F@2 — the largest single trained-model gain, and it did NOT lower val_mse**
(0.0508→0.0524, slightly worse). Geometry fixed the error *structure* (moved it out of the
decoder-sensitive directions), not its magnitude — independent confirmation of §2.

### 3b. Step 2 — DA3 GT-free geometry — **DOES NOT TRANSFER (verified NOT a bug)**
Replace the GT cloud with DA3-predicted depth points (`scripts/cache_da3_points.py`;
`--geom-source da3`). Best val_mse of all runs (0.0496) yet stitch **FURN F@2 ≈ 0.22–0.24** —
*worse than the no-geometry baseline*.

The user (rightly) suspected a bug. Direct tests ruled every one out:
- **Geometry function correct:** `da3_window_geom` reproduces the cached `da3_pts_norm` to NN = 0.0005.
- **Training cache clean:** `da3_pts_norm` aligned across all rooms (NN 0.03–0.06, flips correct).
- **Model healthy:** on cache windows with training geometry it decodes ~6 cm to oracle (≈ baseline).
- **Not frame count:** matching the training frame count at eval (16-view stitch) still collapses (0.221).

| DA3-geom eval variant | FURN F@2 |
|---|---|
| geom-source da3 (8-view) | 0.237 |
| + FPS-matched subsample | 0.234 |
| + 16-frame geometry | 0.233 |
| + 16-view stitch (= training frames) | 0.221 |

**Diagnosis (a real train→deploy gap, not code):** DA3 geometry and DA3 image-tokens are correlated,
so the model learns to **over-rely on the geometry stream**. That stream is high-quality on the
dense 12–24-frame *training* windows but degrades on the sparse 8-frame *deployment* windows, and on
windows where DA3 is locally wrong the model amplifies the error (per-window `pred→GT` spikes to
0.15 m) while baseline / clean-geometry are unaffected.

**Clean science across the three:** `pts_norm` (clean) **+0.032**; DA3 (GT-free) **−0.13**. Geometry
conditioning is only as good as the geometry. **Fix (not yet run): geometry dropout** — randomly
hide the geom stream during training so it can't over-rely; and/or clean the DA3 cloud (RANSAC/conf).

---

## 4. Point-reconstruction loss (endpoint matching) — "Option B"
**Idea:** a correspondence-based, per-point loss instead of set losses (Chamfer/F). Integrate the
frozen decoder's ODE for `z_pred` and `z*` **from the same noise init**, then L2 the final points
**per point** — the shared noise gives correspondence for free ("the point born at noise nᵢ must land
exactly where the oracle sends it"). It is the *full* version of the velocity loss (velocity
constrains the path; endpoint constrains the destination). `endpoint_match_loss` +
`--endpoint-loss-weight/points/steps` (5-step Euler, 256 pts, decoder frozen).

| model | whole Chamfer | whole F@5 | **FURN F@2** | FURN F@5 |
|---|---|---|---|---|
| baseline (plain + vel0.3) | 0.0567 | 0.707 | 0.371 | 0.667 |
| plain + endpoint@0.1 (no vel) | 0.0576 | 0.686 | 0.378 | 0.668 |
| plain + vel0.3 + endpoint@1.0 | 0.0556 (best) | 0.697 | **0.326** ↓ | 0.608 |

- **endpoint@0.1:** +0.007 furniture (small). The endpoint term decayed to negligible by
  mid-training — weight too low to stay influential.
- **endpoint@1.0 (combo):** furniture **drops to 0.326**, while whole Chamfer becomes the **best of
  all (0.0556)**.

**Diagnosis:** the plain per-point endpoint L2 samples **uniform** noise → endpoints land mostly on
the **bulk** surfaces (walls/floor, most of the geometry). So a strong endpoint weight optimizes bulk
hard (best whole Chamfer) while the furniture (thin, a tiny fraction of points) gets *relatively
less* gradient and degrades. **Same bulk-domination as Chamfer** — the exact failure mode we set out
to avoid, now confirmed empirically:

| geometric loss | furniture effect |
|---|---|
| velocity (path) | +0.013 |
| endpoint@0.1 (weak) | +0.007 |
| endpoint@1.0 (strong) | **−0.045** |

**Lesson:** plain geometric losses are **bulk-weighted** — more of them polishes walls, not chairs.
The fix (next) is a **detail-weighted** loss: weight the per-point endpoint L2 by the furniture-region
mask, or use a differentiable **soft-F@2cm** (threshold-focused), so the strong gradient lands on the
furniture rather than the floor.

---

## 5. Synthesis & where the headroom is
- **Loss shape has small leverage** on the current adapter: velocity +0.013, endpoint@0.1 +0.007,
  endpoint@1.0 −0.045. Plain geometric losses plateau or hurt because they're bulk-dominated.
- **Information has the big leverage:** clean geometry +0.032 (largest trained gain), data scaling
  +0.034 (0.337→0.371). But clean geometry is GT-derived; DA3's GT-free version doesn't transfer yet.
- **token-MSE ⟂ detail** is now confirmed four ways: spanshort (best val_mse, worst furniture);
  geometry (better furniture, *worse* val_mse); DA3-geom (best val_mse, worst stitch); endpoint
  (val_mse flat while the loss drops 130×). **Do not use val_mse to judge detail — only the stitch
  furniture metric.**

**Ranked next steps:**
1. **Detail-weighted endpoint / soft-F loss** — the one change that could make "make points land
   exactly" actually help furniture instead of trading it for bulk.
2. **Geometry-dropout retrain** — to make DA3 (GT-free) geometry deployable and recover part of the
   +0.032 clean-geometry gain.
3. **Better geometry recovery** (multi-view DA3 fusion / RANSAC-clean the cloud) — the honest path
   toward the (GT-free) ceiling, since reaching *true* oracle requires GT-quality geometry.

---

## 6. Methodology notes (for the write-up)
- **The eval is a GT-conditioned upper bound, not a deployable number** — and that's a deliberate
  *component-isolation* choice, not a bug. GT is used for: stitching **poses**, the per-window
  **scale `nf`**, and the **training target** `z* = encode(GT visible geometry)`. Real components
  (estimated poses, DA3 geometry) only add error ⇒ the reported numbers upper-bound a fully GT-free
  pipeline. It is *not* an upper bound on the adapter itself (that part is measured honestly).
- **What `pts_norm` is / isn't:** it is the normalized visible cloud that (Role 1) **creates the
  target `z*`** in every experiment, and (Role 3) is **fed into the adapter only in the geometry-
  conditioning experiments**. It is **NOT** used as decoder conditioning — the decode call passes
  `pointmaps=pts_norm` but the ODE wrapper ignores it; the decoder conditions only on the tokens.
- **F@1cm:** added to all tables. **FURN F@1cm is discriminative** (amplifies gaps; oracle 0.240,
  best model 0.140 = only 58% of oracle vs 67% at F@2 — finest scale has the most headroom).
  **Whole F@1cm is near-floor noise** — the oracle scores *below* the models on it — because the 1 cm
  threshold is below both the decode scatter (~3 cm median) and the point spacing (~1.4 cm @200k), so
  it measures sampling/discretization, not accuracy. Report it as "raw" but compare on FURN F@1 / F@5.

---

## 7. Key files (later work)
- `scripts/token_noise_sensitivity.py` — §2 sensitivity sweep (+ `token_sensitivity.csv`).
- `scripts/cache_da3_points.py` — DA3 GT-free geometry cache (§3b); `da3_window_geom` in stitch.
- `scripts/ransac_drop_eval.py` — §1 window drop-k re-eval.
- `train_online_var_hungarian.py` — `--geom-cond/--geom-source` (§3), `endpoint_match_loss` +
  `--endpoint-loss-weight/points/steps` (§4).
- `stitch_office4.py` — `--geom-source {pts_norm,da3}`, `--geom-frames` (geom-aware eval).
- Checkpoints: `fullrec_nf16_vel03_geom_best.pt` (pts_norm-geom), `..._da3geom_best.pt` (DA3-geom),
  `fullrec_nf16_endpoint_best.pt` (endpoint@0.1), `fullrec_nf16_endcombo_best.pt` (combo).
