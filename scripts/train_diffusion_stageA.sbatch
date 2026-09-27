#!/bin/bash
#SBATCH --job-name=diff_stageA
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
# Best-GPU LIST (hard): only bf16-capable cards — H100(CC9.0), RTX PRO 6000 Blackwell(12.0),
# RTX 6000 Ada / L40S(8.9), A40 / A6000(8.6). Never a Turing/Pascal card (no bf16).
# HARD OR-list of every bf16-capable card we can actually run on: H100(9.0), Ada/L40S(8.9),
# A40/A6000(8.6). Exclude Blackwell(12.0) — torch 2.2/cu121 maxes at sm_90 and would crash sm_120.
# This lets the job FALL BACK to an A40 if H100s are reserved, instead of idling for days.
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
# Soft preference: still grab the fast H100 tier first when one is free.
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_stageA_%j.log
# Stage A: DiffComplete DDPM pretraining on 3D-FRONT room crops. Resumable; re-globs the cache
# at every (re)start, so a later restart automatically picks up more cached rooms.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_front3d \
  --steps 200000 --batch-size 4 --lr 1e-4 --warmup 5000 \
  --max-scenes 1600 --workers 8 \
  --eval-every 5000 --ckpt-every 2000
echo "STAGE A DONE"
