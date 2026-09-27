#!/bin/bash
#SBATCH --job-name=occfill
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/occfill_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/occfill_%j.log
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID"
srun python scripts/train_occ_filler.py --steps 8000 --batch-size 32 \
  --hidden 384 --depth 6 --eval-every 2000 --val-windows 8
echo "[sbatch] DONE"
