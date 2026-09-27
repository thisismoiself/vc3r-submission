#!/bin/bash
#SBATCH --job-name=pf_one
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/pf_one_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/pf_one_%j.log

# Single-config office4 stitch eval (midpoint, 50k, base decoder, fullrec adapter).
# Args: $1 = corrector ckpt path, $2 = out-tag, $3 = corrector-steps
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID corr=$1 tag=$2 steps=$3"

srun python scripts/stitch_office4.py \
  --ckpt outputs/consecutive_windows/fullrec_nf16_vel03_complete_best.pt \
  --complete-target --fm-sampling midpoint --num-queries 50000 \
  --point-corrector "$1" --corrector-steps "$3" --out-tag "$2"

echo "[sbatch] DONE job=$SLURM_JOB_ID"
