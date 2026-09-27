# Results while away

_updated 2026-07-18 19:04:21_

## Reference
- baseline (Replica-only, no corrector): whole chamfer 0.0567 | FURN F@2 0.3709 F@5 0.667
- oracle ceiling:                          whole chamfer 0.0440 | FURN F@2 0.5510 F@5 0.807

## 1. Point-flow corrector on office4 (your point-space FM idea)
```
BASELINE (this run, no corrector, seed 42):
  PRED   : whole chamfer 0.0567  F@5 0.707  | FURN F@2 0.3709  F@5 0.6666
  ORACLE : whole chamfer 0.0446  F@5 0.812  | FURN F@2 0.5512  F@5 0.8068


CORRECTED (this run, seed 42):
  PRED   : whole chamfer 0.0545  F@5 0.721  | FURN F@2 0.3644  F@5 0.6501
  ORACLE : whole chamfer 0.0450  F@5 0.811  | FURN F@2 0.5447  F@5 0.8061

```
CONTROLLED DELTA (corrected - baseline, same seed):
  whole chamfer -0.0021  whole F@5 +0.014  | FURN F@2 -0.0065  FURN F@5 -0.0165
  (FURN F@2 > 0 => corrector helps furniture detail; < 0 => bulk-smoother hurts it)
STATUS: corrected DONE
  baseline DONE

Read: the BASELINE above is the SAME run/seed with no corrector -> apples-to-apples,
free of the ~0.006 cross-run furniture noise. Whole-scene already improved (chamfer,
F@5); the controlled furniture delta below is the real verdict.

## 2. NRGBD data-scaling on office4 (Replica-7 + NRGBD-9, val=office4)
```
NRGBD-DATA:
  PRED   : whole chamfer 0.0534  F@5 0.739  | FURN F@2 0.3815  F@5 0.6786
  ORACLE : whole chamfer 0.0436  F@5 0.828  | FURN F@2 0.5600  F@5 0.8161

```
STATUS: DONE

Read: FURN F@2 vs baseline 0.3709 (did +9 scenes of data help office4 detail?).

## Raw logs
- point-flow stitch: experiments/overfit_8frames/eval_pf_corrected_stitch.log
- point-flow train:  experiments/overfit_8frames/point_flow_full.log
- NRGBD train:       experiments/overfit_8frames/fullrec_nrgbd.log
- NRGBD stitch:      experiments/overfit_8frames/eval_fullrec_nrgbd_stitch.log