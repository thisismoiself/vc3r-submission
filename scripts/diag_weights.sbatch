#!/bin/bash
#SBATCH --job-name=diag_w
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diag_w_%j.log
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/diag_weights.py --room-idxs 1 2 3
echo "DIAG DONE"
