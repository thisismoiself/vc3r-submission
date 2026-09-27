#!/bin/bash
#SBATCH --job-name=eval_rooms
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/eval_rooms_%j.log
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/eval_rooms.py --room-idxs 1 2 3 4 5 6 --steps 50
echo "EVAL DONE"
