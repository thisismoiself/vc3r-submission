#!/usr/bin/env bash
#SBATCH --job-name=ckpt_sample_eval
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/eval_niko/ckpt_sample_eval_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/eval_niko/ckpt_sample_eval_%j.out

set -euo pipefail
cd /usr/prakt/s0017/vc3r
PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
export NOVA3R_DIR=nova3r
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun "$PY" -u scripts/evaluate_checkpoint_sample.py "$@"
