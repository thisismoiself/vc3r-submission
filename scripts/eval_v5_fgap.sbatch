#!/bin/bash
#SBATCH --job-name=v5_fgap
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/v5_fgap_%j.log
# Backfill the periodic strip-F for v5 ladder checkpoints that went unscored during the
# maintenance drain (152k-182k, every 4k). Same eval as eval_v5_ckpt.sbatch (5-crop strip,
# track-rooms 0-4, 100 DDIM steps). Writes best-ckpt/json + strips to a SCRATCH dir so it does
# NOT clobber the live stageA_v5_periodicBEST.{pt,json}.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
SNAP=outputs/consecutive_windows/snapshots
SCR=experiments/overfit_8frames/fgap_scratch
mkdir -p "$SCR"
echo "=== v5 strip-F backfill (unscored ladder ckpts) ==="
for s in 152000 156000 160000 164000 168000 172000 176000 180000 182000; do
  CK=$SNAP/stageA_v5_ladder_${s}.pt
  [ -f "$CK" ] || { echo "F @ step $s = MISSING_CKPT"; continue; }
  F=$($PY scripts/progress_strip.py --band 0.06 --ckpt "$CK" \
        --cache-dir outputs/tsdf_cache_front3d_v2_holdout --val-frac 1.0 \
        --room-idx 0 --track-rooms 0 1 2 3 4 --steps 100 \
        --best-ckpt "$SCR/gap_best.pt" --best-json "$SCR/gap_best.json" \
        --out-dir "$SCR/gap_strip" 2>/dev/null | grep -oE "cross-room F=[0-9.]+" | tail -1)
  echo "F @ step $s = ${F:-ERR}"
done
echo "FGAP DONE"
