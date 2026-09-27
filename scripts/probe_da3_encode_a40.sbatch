#!/bin/bash
#SBATCH --job-name=da3enc
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/da3enc_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/da3enc_%j.log
set -euo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID"
srun python scripts/probe_da3_encode.py
echo "[sbatch] DONE"
