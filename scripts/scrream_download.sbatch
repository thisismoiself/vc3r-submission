#!/bin/bash
#SBATCH --job-name=scrream_dl
#SBATCH --partition=NORMAL
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=42:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/scrream_dl_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/scrream_dl_%j.log
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[dl] host=$(hostname) job=$SLURM_JOB_ID $(date)"
curl -sI https://drive.google.com >/dev/null 2>&1 && echo "[dl] internet OK" || echo "[dl] NO INTERNET on this node"
python /usr/prakt/s0016/SCRREAM/orchestrate.py
echo "[dl] DONE $(date)"
