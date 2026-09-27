#!/usr/bin/env bash
#SBATCH --job-name=velloss
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/velloss_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/velloss_%j.out
# Velocity-matching aux loss: plain token-MSE + lambda * velocity-match, batch 24,
# on the run1 cache (700 windows) so it is a direct head-to-head with abl_plain_best.pt.
# Two lambda values to bracket the weight.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/run1_span24_100_nf4_10_l13
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for W in 1.0 0.1; do
  TAG=$(echo "$W" | tr -d '.')
  echo "=== [$(date)] TRAIN plain+vel lambda=$W ==="
  $PY scripts/train_online_var_hungarian.py \
    --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
    --val-rooms $C/office4 --max-per-root 100 --max-val-per-root 60 \
    --steps 12000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
    --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
    --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
    --loss-mode plain --vel-loss-weight "$W" --vel-points 1024 \
    --ckpt-out "$OUT/vel_lambda${TAG}_best.pt" \
    > experiments/overfit_8frames/vel_lambda${TAG}.log 2>&1
  echo "=== [$(date)] DONE lambda=$W exit=$? ==="
done
echo "=== [$(date)] VELOCITY RUNS COMPLETE ==="
