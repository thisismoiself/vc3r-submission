#!/bin/bash
#SBATCH --job-name=prog_strip
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
# Any bf16-or-fp16 GPU is fine for 50-step DDIM sampling; PREFER a weak card so we never steal the
# fast training tier. No hard constraint -> scheduler can place it anywhere with a free GPU.
#SBATCH --prefer="GPU_CC:7.5"
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/prog_strip_%j.log
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
# Seed an early frame from the step-16k snapshot, render the live checkpoint, then WATCH and append
# a new frame each 5k-step milestone.
$PY scripts/progress_strip.py \
  --extra-ckpts /usr/prakt/s0016/vc3r/outputs/consecutive_windows/snapshots/stageA_step16k.pt \
  --milestone 5000 --steps 50 --poll-min 15 --watch
echo "STRIP DONE"
