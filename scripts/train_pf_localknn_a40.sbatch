#!/bin/bash
#SBATCH --job-name=pf_lknn
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/pf_lknn_a40_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/pf_lknn_a40_%j.log

# Stage-2 point-flow corrector with LOCAL GEOMETRY CONDITIONING (k-NN PointNet) + scaled
# capacity. Decoder is untouched (frozen NOVA3R); this only changes the corrector network.
# Reuses the existing P_pred bank (base adapter fullrec_nf16_vel03_complete_best, q8192).
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CACHE=scripts/data/fc_nf16_span24_100_l13_complete

srun python scripts/train_point_flow.py \
  --base-ckpt outputs/consecutive_windows/fullrec_nf16_vel03_complete_best.pt \
  --rooms $CACHE/office0 $CACHE/office1 $CACHE/office2 $CACHE/office3 $CACHE/room0 $CACHE/room1 $CACHE/room2 \
  --val-rooms $CACHE/office4 \
  --bank-dir outputs/point_flow_bank_complete --bank-queries 8192 \
  --local-knn 16 --hidden 512 --depth 8 \
  --steps 16000 --batch-size 8 --n-couple 1024 --lr 3e-4 \
  --eval-every 2000 --eval-windows 8 \
  --ckpt-out outputs/consecutive_windows/point_flow_localknn16_h512.pt

echo "[sbatch] DONE job=$SLURM_JOB_ID"
