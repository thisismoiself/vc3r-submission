#!/bin/bash
#SBATCH --job-name=recache_da3pose
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/recache_da3pose_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/recache_da3pose_%j.log
# Regenerate da3_tokens.pt for all Replica windows with the GT-FREE camera path
# (cam_token=None). Reuses z*/pts_norm/meta (GT complete target) by symlink. Resumable.
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 DA3_LOG_LEVEL=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python scripts/recache_gtfree_tokens.py \
  --src-cache scripts/data/fc_nf16_span24_100_l13_complete \
  --dst-cache scripts/data/fc_nf16_span24_100_l13_complete_da3pose \
  --pose-source da3pred --replica-root /usr/prakt/s0016/Replica --rooms office0 office1 office2 office3 room0 room1 room2 office4
echo "[sbatch] RECACHE DONE job=$SLURM_JOB_ID"
