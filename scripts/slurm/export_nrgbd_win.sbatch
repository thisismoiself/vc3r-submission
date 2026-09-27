#!/usr/bin/env bash
#SBATCH --job-name=nrgbd_win
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/nrgbd_win_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/nrgbd_win_%j.out
set -uo pipefail
cd /usr/prakt/s0016/vc3r
export NOVA3R_DIR=nova3r_lib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python -u scripts/export_nrgbd_window.py
