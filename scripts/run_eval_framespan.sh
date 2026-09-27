#!/usr/bin/env bash
# Fair eval of frame/span checkpoints: identical office4 deployment (8-frame, stride 10),
# midpoint stitch 50k + furniture F-scores + alignment diagnostic. val_mse is NOT
# comparable across caches (different val sets); this stitch IS.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/consecutive_windows

for L in "$@"; do
  CK=$OUT/fs_${L}_best.pt
  TAG=fs_${L}_midpoint
  echo "=== [$(date)] STITCH $L ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 --stride 10 \
    --num-queries 50000 --fm-sampling midpoint --out-tag "$TAG" \
    > experiments/overfit_8frames/eval_fs_${L}_stitch.log 2>&1
  echo "  stitch exit=$?"
  $PY scripts/align_diagnostic.py "outputs/replica/stitch_office4_${TAG}/per_window.npz" \
    > experiments/overfit_8frames/eval_fs_${L}_align.log 2>&1
  echo "  align exit=$?"
done
echo "=== [$(date)] FRAMESPAN EVAL COMPLETE ==="
