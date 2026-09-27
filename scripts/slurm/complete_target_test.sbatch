#!/usr/bin/env bash
#SBATCH --job-name=comptest
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/comptest_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/comptest_%j.out
set -uo pipefail
cd /usr/prakt/s0016/vc3r
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python -u scripts/tmp_complete_target_test.py
echo "=== [$(date)] DONE exit=$? ==="
