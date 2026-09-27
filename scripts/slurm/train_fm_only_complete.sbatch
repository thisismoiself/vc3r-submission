#!/usr/bin/env bash
# PURE FLOW-MATCHING adapter training on the COMPLETE-geometry tokens (Replica, office4 held out).
# Recipe as train_fullrecipe.sbatch (nf16 complete cache, b24, 18k = 2 cosine cycles, EMA,
# drop/tokdrop 0.2) EXCEPT: loss is FM-only, and windows are capped at 60/room (420 train) so
# the non-lazy DA3 cache fits in RAM on this 31GB box (the full 1225-window DA3 cache is ~16-18GB
# float16 -> too big for both /tmp prestage and float32 RAM here). Per-step time is independent
# of window count, so measured s/it is representative of the full run. FM-only changes:
#   --mse-weight 0  (turn off the Hungarian token-MSE base loss)
#   --vel-loss-weight 1.0  (velocity-matching on NOVA3R's cosine FM path = the ONLY signal)
# NOTE: with mse-weight 0, token-MSE (val_w/val_mse) is NOT the objective and will RISE during
# training -> the val_w-based *_best.pt is misleading. Evaluate *_last.pt geometrically instead.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13_complete
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 60 --max-val-per-root 40 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 100 --ckpt-every 1000 \
  --loss-mode plain --mse-weight 0 --vel-loss-weight 1.0 --vel-points 1024 \
  --ckpt-out "$OUT/fm_only_complete_best.pt"
echo "=== [$(date)] FM-ONLY COMPLETE DONE exit=$? ==="
