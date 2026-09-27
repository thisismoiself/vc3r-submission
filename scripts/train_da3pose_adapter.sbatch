#!/bin/bash
#SBATCH --job-name=da3pose_adapter
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/da3pose_adapter_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/da3pose_adapter_%j.log
# GT-FREE-INPUT adapter: DA3 tokens from the pose-free path (cam_token=None),
# GT complete-target z*. Recipe = complete baseline (plain + vel0.3, nf16 12-24 views,
# 1225 windows, complete target, 18k steps). PREEMPTION-SAFE: auto-resumes from the
# latest *_step<N>.pt snapshot (--init-ckpt restores weights+scheduler+step), so a
# requeue after preemption continues instead of restarting.
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CACHE=scripts/data/fc_nf16_span24_100_l13_complete_da3pose
CKPT=outputs/consecutive_windows/da3pose_nf16_vel03_complete_best.pt
TOTAL=18000

# pick the highest-step snapshot to resume from (empty on a fresh run)
LATEST=$(ls -1 outputs/consecutive_windows/da3pose_nf16_vel03_complete_best_step*.pt 2>/dev/null \
         | sed -E 's/.*_step([0-9]+)\.pt/\1 &/' | sort -n | tail -1 || true)
STEP=$(echo "$LATEST" | awk '{print $1}')
SNAP=$(echo "$LATEST" | awk '{print $2}')
INIT=(); STEPS=$TOTAL
if [ -n "${STEP:-}" ] && [ "$STEP" -lt "$TOTAL" ]; then
  INIT=(--init-ckpt "$SNAP"); STEPS=$((TOTAL - STEP))
  echo "[resume] from $SNAP (step $STEP) -> $STEPS steps remaining"
else
  echo "[fresh] $TOTAL steps"
fi

srun python scripts/train_online_var_hungarian.py \
  --rooms $CACHE/office0 $CACHE/office1 $CACHE/office2 $CACHE/office3 $CACHE/room0 $CACHE/room1 $CACHE/room2 \
  --val-root $CACHE/office4 \
  --loss-mode plain \
  --vel-loss-weight 0.3 --vel-points 1024 \
  --drop 0.2 --token-drop 0.2 --batch-size 24 \
  --steps "$STEPS" --warmup-steps 1000 --lr 5e-4 --lr-min 1e-6 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 --weight-decay 1e-4 \
  --ckpt-every 2000 "${INIT[@]}" \
  --ckpt-out "$CKPT"
echo "[sbatch] TRAIN DONE job=$SLURM_JOB_ID"
