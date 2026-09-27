#!/bin/bash
#SBATCH --job-name=tsdf_s1
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/tsdf_stage1_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/tsdf_stage1_%j.log
set -uo pipefail
cd /usr/prakt/s0016/vc3r
source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r
echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 1) ensure caches exist (skips already-cached scenes; harmless if workstation already did them)
python scripts/cache_all.py
# 2) train Stage 1 (resumable: re-runs continue from the checkpoint)
python scripts/train_stage1.py --steps 80000 --batch-size 4 --workers 6 \
  --val-scenes scene11 office4 --eval-every 4000 --ckpt-every 4000 \
  --ckpt /usr/prakt/s0016/vc3r/outputs/consecutive_windows/stage1_scrream.pt
echo "[sbatch] DONE $(date)"
