#!/bin/bash
#SBATCH --job-name=diff_gate
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_overfit_gate_%j.log
# Build-order gate (spec step 3): overfit ONE 3D-FRONT crop, DDIM-sample it back.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py \
  --overfit "outputs/tsdf_cache_front3d/00004f89-9aa5-43c2-ae3c-129586be8aaa_Library-4425.npz" \
  --steps 2000 --sample-every 500 \
  --viz-out outputs/consecutive_windows/diff_overfit_completion.npz
echo "GATE DONE"
