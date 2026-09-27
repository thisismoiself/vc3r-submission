#!/bin/bash
#SBATCH --job-name=recache_gtfree_nrgbd
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/recache_gtfree_nrgbd_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/recache_gtfree_nrgbd_%j.log
# GT-free da3_tokens for NeuralRGBD windows (cam_token=None), reuse z*/pts_norm/meta
# (GT complete target) by symlink. Resumable (skips existing). NRGBD uses the same
# results/frameNNNNNN.jpg layout as Replica.
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 DA3_LOG_LEVEL=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python scripts/recache_gtfree_tokens.py \
  --src-cache scripts/data/nrgbd_nf16_span24_100_l13_complete \
  --dst-cache scripts/data/nrgbd_nf16_span24_100_l13_complete_gtfree \
  --replica-root /usr/prakt/s0016/NeuralRGBD \
  --rooms complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom breakfast_room
echo "[sbatch] RECACHE NRGBD DONE job=$SLURM_JOB_ID"
