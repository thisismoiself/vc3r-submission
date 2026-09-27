#!/usr/bin/env bash
#SBATCH --job-name=fullrec
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_%j.out
# FULL RECIPE on scaled data: nf16 distribution (1225 train windows) + plain loss
# + velocity aux (lambda 0.3) + midpoint eval. Longer schedule (18k = 2 cosine cycles).
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --ckpt-out "$OUT/fullrec_nf16_vel03_best.pt" \
  > experiments/overfit_8frames/fullrec_nf16_vel03.log 2>&1
echo "=== [$(date)] FULL RECIPE DONE exit=$? ==="
