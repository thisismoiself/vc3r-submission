#!/usr/bin/env bash
# SPAN axis: hold n_frames=8 (deployment-matched), vary span to probe DETAIL.
# Waits for the frame-count caching driver to finish first (single GPU -> serialize).
# Combined with the nf8 anchor (span 24-100) this gives a span/detail trend:
#   short  : span 16-36  (high token density -> finest detail)
#   anchor : span 24-100 (nf8 cache, built by run_framecount_caches.sh)
#   long   : span 120-300 (low density -> coarse; completeness ceiling)
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
FCLOG=experiments/overfit_8frames/cache_framecount.log

echo "=== [$(date)] waiting for frame-count caching to finish ==="
until grep -q "CACHING DONE" "$FCLOG" 2>/dev/null; do sleep 60; done
echo "=== [$(date)] frame-count caching done; starting span caches ==="

COMMON="--replica-root /usr/prakt/s0016/Replica --all-scenes --windows-per-room 70 \
  --n-frames-min 8 --n-frames-max 9 --n-seeds 30 --da3-layer 1 3 --da3-max-tokens 2048 \
  --image-height 392 --image-width 518 --flip"

echo "=== [$(date)] CACHE span_short (16-36, nf8) ==="
$PY scripts/cache_online_hungarian_zstar_windows.py $COMMON \
  --span-min 16 --span-max 36 \
  --out-root scripts/data/fc_spanshort_16_36_nf8_l13

echo "=== [$(date)] CACHE span_long (120-300, nf8) ==="
$PY scripts/cache_online_hungarian_zstar_windows.py $COMMON \
  --span-min 120 --span-max 300 \
  --out-root scripts/data/fc_spanlong_120_300_nf8_l13

echo "=== [$(date)] SPAN CACHING DONE ==="
