#!/usr/bin/env bash
# MATCHED MSE baseline for the pure-FM run: IDENTICAL data (same first-60/room = 420 windows,
# deterministic w[:60] slice), same seed (42), same schedule (b24, 18k=2 cosine cycles, EMA,
# drop/tokdrop 0.2, ckpt-every 1000) as train_fm_only_complete.sbatch -- the ONLY difference is the
# loss: plain Hungarian-MSE (--mse-weight 1 default, --vel-loss-weight 0), no flow-matching term.
# This isolates the loss so FM-420 vs MSE-420 is a fair equal-data comparison. MSE-only has no
# decoder in the loop -> faster (~0.3-0.7 s/it) and lower memory than the FM run.
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
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 --seed 42 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 100 --ckpt-every 1000 \
  --loss-mode plain --mse-weight 1 --vel-loss-weight 0 \
  --ckpt-out "$OUT/mse420_complete_best.pt"
echo "=== [$(date)] MSE-420 COMPLETE DONE exit=$? ==="
