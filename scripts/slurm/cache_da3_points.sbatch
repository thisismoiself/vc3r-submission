#!/usr/bin/env bash
#SBATCH --job-name=da3pts
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/da3pts_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/da3pts_%j.out
# STEP 2 prerequisite: DA3-predicted geometry per window (GT-free), written as
# da3_pts_norm.pt beside each nf16 cache window, in the pts_norm frame. Idempotent
# (skips windows already done). Frame validated: da3->pts_norm NN ~0.05 norm-units.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [$(date)] host=$(hostname) ==="
$PY scripts/cache_da3_points.py \
  --cache-root scripts/data/fc_nf16_span24_100_l13 --fps \
  > experiments/overfit_8frames/cache_da3_points.log 2>&1
echo "=== [$(date)] DA3-POINTS CACHE DONE exit=$? ==="
grep -c da3_pts_norm scripts/data/fc_nf16_span24_100_l13/*/*/da3_pts_norm.pt 2>/dev/null | wc -l
