#!/usr/bin/env bash
#SBATCH --job-name=lossabl
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/lossabl_%A_%a.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/lossabl_%A_%a.out
# Phase 0: 4-arm loss ablation, one --loss-mode per array task, each on a dedicated GPU.
# Identical data (run1 cache, 7 train rooms, LOO office4) + run2 hyperparameters.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODES=(plain sqrt_var var_weighted huber)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
C=scripts/data/run1_span24_100_nf4_10_l13
OUT=outputs/consecutive_windows
mkdir -p "$OUT"

echo "=== [$(date)] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES loss=$MODE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 100 --max-val-per-root 60 \
  --steps 12000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode "$MODE" \
  --ckpt-out "$OUT/abl_${MODE}_best.pt"
echo "=== [$(date)] DONE loss=$MODE exit=$? ==="
