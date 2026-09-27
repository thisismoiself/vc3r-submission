#!/bin/bash
#SBATCH --job-name=v5_pe
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/v5_pe_%j.log
# Periodic one-shot held-out F for a given v4 checkpoint snapshot ($1). Comparable to v3 BEST F=83.9.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
$PY scripts/progress_strip.py --band 0.06 --ckpt "$1" \
  --cache-dir outputs/tsdf_cache_front3d_v2_holdout --val-frac 1.0 \
  --room-idx 0 --track-rooms 0 1 2 3 4 --steps 100 \
  --best-ckpt outputs/consecutive_windows/snapshots/stageA_v5_periodicBEST.pt \
  --best-json outputs/consecutive_windows/snapshots/stageA_v5_periodicBEST.json \
  --out-dir experiments/overfit_8frames/progress_strip_v4_periodic
echo "PE DONE"
