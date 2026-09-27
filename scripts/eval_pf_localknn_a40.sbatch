#!/bin/bash
#SBATCH --job-name=pf_eval
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/pf_eval_a40_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/pf_eval_a40_%j.log

# Definitive office4 stitch eval (midpoint, 50k queries, base decoder, fullrec adapter):
#   (A) NEW local-knn corrector, sweep step counts -> its FURN F@2
#   (B) OLD corrector, steps 6 -> reproduces the 0.356 reference in THIS environment
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader

BASE=outputs/consecutive_windows/fullrec_nf16_vel03_complete_best.pt
COMMON="--ckpt $BASE --complete-target --fm-sampling midpoint --num-queries 50000"

echo "########## A: NEW local-knn corrector (steps sweep) ##########"
srun python scripts/stitch_office4.py $COMMON \
  --point-corrector outputs/consecutive_windows/point_flow_localknn16_h512.pt \
  --corrector-steps-sweep 0,6,12,20 --out-tag pf_localknn_q50

echo "########## B: OLD corrector steps=6 (0.356 reference) ##########"
srun python scripts/stitch_office4.py $COMMON \
  --point-corrector outputs/consecutive_windows/point_flow_complete.pt \
  --corrector-steps 6 --out-tag pf_oldcorr_ref_q50

echo "[sbatch] DONE job=$SLURM_JOB_ID"
