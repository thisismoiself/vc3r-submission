#!/bin/bash
#SBATCH --job-name=abl_ct
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/abl_ct_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/abl_ct_%j.log

# COMPLETE-TARGET loss-lever ablation. Replicates the base complete recipe
# (fullrec_nf16_vel03_complete = plain + vel0.3, log line 73) EXACTLY, varying only
# --loss-mode and --vel-loss-weight. No --pca-project (known-negative on complete),
# no --decoder-head (base used the plain NOVA3R checkpoint-last.pth). Tokens held in RAM.
#   Args: $1=loss-mode (plain|var_weighted)  $2=vel-loss-weight  $3=tag
set -euo pipefail
LOSS="$1"; VEL="$2"; TAG="$3"

REPO=/usr/prakt/s0016/vc3r
cd "$REPO"
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES loss=$LOSS vel=$VEL tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CACHE=scripts/data/fc_nf16_span24_100_l13_complete

srun python scripts/train_online_var_hungarian.py \
  --rooms $CACHE/office0 $CACHE/office1 $CACHE/office2 $CACHE/office3 $CACHE/room0 $CACHE/room1 $CACHE/room2 \
  --val-root $CACHE/office4 \
  --loss-mode "$LOSS" \
  --vel-loss-weight "$VEL" --vel-points 1024 \
  --drop 0.2 --token-drop 0.2 --batch-size 24 \
  --steps 18000 --warmup-steps 1000 --lr 5e-4 --lr-min 1e-6 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 --weight-decay 1e-4 \
  --ckpt-every 2000 \
  --ckpt-out outputs/consecutive_windows/abl_complete_${TAG}_best.pt

echo "[sbatch] DONE job=$SLURM_JOB_ID tag=$TAG"
