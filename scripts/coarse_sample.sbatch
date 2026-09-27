#!/bin/bash
#SBATCH --job-name=coarse_smp
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/coarse_smp_%j.log
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python scripts/coarse_sample.py
echo DONE
