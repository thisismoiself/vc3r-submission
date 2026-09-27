#!/usr/bin/env bash
# Evaluate loss-ablation checkpoints on office4: midpoint stitch (50k) + furniture
# F-scores + alignment diagnostic. Runs on the login GPU (CPU-light decode fits).
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/consecutive_windows

for M in plain var_weighted huber; do
  CK=$OUT/abl_${M}_best.pt
  TAG=abl_${M}_midpoint
  echo "=== [$(date)] STITCH $M ($CK) ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 --stride 10 \
    --num-queries 50000 --fm-sampling midpoint --out-tag "$TAG" \
    > experiments/overfit_8frames/eval_${M}_stitch.log 2>&1
  echo "  stitch exit=$?"
  echo "=== [$(date)] ALIGN-DIAG $M ==="
  $PY scripts/align_diagnostic.py "outputs/replica/stitch_office4_${TAG}/per_window.npz" \
    > experiments/overfit_8frames/eval_${M}_align.log 2>&1
  echo "  align exit=$?"
done
echo "=== [$(date)] LOSS EVAL COMPLETE ==="
