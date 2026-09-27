#!/usr/bin/env bash
# Phase 0: 4-arm loss ablation on the EXISTING run1 cache (nf 4-10, span 24-100).
# Identical data + hyperparameters across arms; only --loss-mode changes.
# office4 held out as LOO val. Each arm -> its own best checkpoint.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # shared GPU; reduce fragmentation
C=scripts/data/run1_span24_100_nf4_10_l13
OUT=outputs/consecutive_windows
LOG=experiments/overfit_8frames

COMMON="--rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 100 --max-val-per-root 60 \
  --lazy-da3 --da3-cache-dir /tmp/claude-264016/da3_f16_cache --lru-windows 300 \
  --steps 12000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200"

for MODE in plain sqrt_var var_weighted huber; do
  echo "=== [$(date)] TRAIN loss=$MODE ==="
  $PY scripts/train_online_var_hungarian.py $COMMON \
    --loss-mode "$MODE" \
    --ckpt-out "$OUT/abl_${MODE}_best.pt" \
    > "$LOG/abl_${MODE}.log" 2>&1
  echo "=== [$(date)] DONE loss=$MODE  exit=$? ==="
done
echo "=== [$(date)] ABLATION COMPLETE ==="
