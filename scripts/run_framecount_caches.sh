#!/usr/bin/env bash
# Phase 1: cache two frame-count distributions, span held at 24-100 (matches run1),
# varying ONLY n_frames. Combined with run1 (nf 4-10) this gives a 3-point frame axis.
set -euo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
COMMON="--replica-root /usr/prakt/s0016/Replica --all-scenes --windows-per-room 70 \
  --span-min 24 --span-max 100 --n-seeds 30 --da3-layer 1 3 --da3-max-tokens 2048 \
  --image-height 392 --image-width 518 --flip"

echo "=== [$(date)] CACHE nf8 (deployment-matched 8-frame) ==="
$PY scripts/cache_online_hungarian_zstar_windows.py $COMMON \
  --n-frames-min 8 --n-frames-max 9 \
  --out-root scripts/data/fc_nf8_span24_100_l13

echo "=== [$(date)] CACHE nf16 (12-24 frames, more completeness) ==="
$PY scripts/cache_online_hungarian_zstar_windows.py $COMMON \
  --n-frames-min 12 --n-frames-max 24 \
  --out-root scripts/data/fc_nf16_span24_100_l13

echo "=== [$(date)] CACHING DONE ==="
