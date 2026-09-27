#!/bin/bash
#SBATCH --job-name=latflow
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/latflow_a40_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/latflow_a40_%j.log

# First experiment: conditional latent flow-matching adapter over z* (samples the latent
# instead of regressing the mean). Held-out office4 head-to-head vs the deterministic adapter.
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader

srun python scripts/train_latent_flow.py \
  --rooms office0 office1 office2 office3 room0 room1 room2 --val-room office4 \
  --det-ckpt outputs/consecutive_windows/complete_bestcombo_pca32_best.pt \
  --steps 6000 --batch-size 16 --lr 3e-4 --hidden 512 --depth 6 \
  --sample-steps 20 --eval-queries 30000 --eval-every 3000 --val-windows 6 \
  --ckpt-out outputs/consecutive_windows/latent_flow_poc.pt

echo "[sbatch] DONE job=$SLURM_JOB_ID"
