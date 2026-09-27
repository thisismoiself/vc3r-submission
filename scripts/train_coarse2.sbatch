#!/bin/bash
#SBATCH --job-name=diff_coarse
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_coarse_%j.log
# Cascade COARSE stage: whole-room completion at 8cm (band 0.24). No cropping, no frontier restriction
# (we WANT global completion of all walls/floor/ceiling). Small grids -> big batch, fast. This model's
# output (upsampled) becomes the global-context conditioning for the fine stage.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py --coarse \
  --cache-dir outputs/tsdf_cache_front3d_coarse --band 0.24 \
  --ckpt outputs/consecutive_windows/diffusion_coarse.pt \
  --surf-w 1.0 --frontier-dilate 0 --cond-drop 0.10 \
  --steps 120000 --batch-size 4 --lr 1e-4 --warmup 3000 \
  --max-scenes 4000 --workers 6 \
  --eval-every 100000 --ckpt-every 2000
echo "COARSE DONE"
