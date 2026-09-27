#!/bin/bash
#SBATCH --job-name=coarse_fd12
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/coarse_fd12_%j.log
# Cascade COARSE stage, STEP-1 fix: v3's proven frontier-restricted loss (--frontier-dilate 12) instead
# of the blob recipe (dilate 0 = grade all unknown -> 85% exterior spray). FROM SCRATCH (new ckpt path)
# so the fill-everywhere habit of the fd0 44k weights can't freeze into the now-ungraded exterior.
# Same clean v2-family cache/band as before; only the loss-scoring region changes.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py --coarse \
  --cache-dir outputs/tsdf_cache_front3d_coarse --band 0.24 \
  --ckpt outputs/consecutive_windows/diffusion_coarse_fd12.pt \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 120000 --batch-size 4 --lr 1e-4 --warmup 3000 \
  --max-scenes 4000 --workers 5 \
  --eval-every 100000 --ckpt-every 2000
echo "COARSE_FD12 DONE"
