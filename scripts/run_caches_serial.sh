#!/usr/bin/env bash
# Serialize all caching AFTER the loss ablation finishes (shared GPU: one heavy job at a time).
# Frame-count caches resume by skipping already-cached windows.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
ABLLOG=experiments/overfit_8frames/loss_ablation_driver.log

echo "=== [$(date)] waiting for loss ablation to complete ==="
until grep -q "ABLATION COMPLETE" "$ABLLOG" 2>/dev/null; do sleep 120; done
echo "=== [$(date)] ablation done; starting caching ==="

bash scripts/run_framecount_caches.sh >> experiments/overfit_8frames/cache_framecount.log 2>&1
bash scripts/run_span_caches.sh        >> experiments/overfit_8frames/cache_span.log 2>&1
echo "=== [$(date)] ALL CACHING DONE ==="
