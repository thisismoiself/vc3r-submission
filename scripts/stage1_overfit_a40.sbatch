#!/bin/bash
#SBATCH --job-name=s1overfit
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stage1_overfit_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stage1_overfit_%j.log
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
srun python scripts/stage1_unet.py --overfit --steps 800 --base 16
echo "[sbatch] DONE"
