#!/bin/bash
#SBATCH --job-name=nrgbd_cache
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/nrgbd_cache_%j.log
# DA3 (LARGE-1.1) reconstruction of the 9 NeuralRGBD scenes (Replica layout) -> band-0.06 DA3 caches
# (partial=DA3 with its holes, GT=clean mesh) so v3 can be tested on a SECOND dataset.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
$PY scripts/replica_da3_cache.py \
  --root /usr/prakt/s0016/NeuralRGBD \
  --rooms breakfast_room complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom \
  --out-dir /usr/prakt/s0016/vc3r/outputs/tsdf_cache_nrgbd_da3_006 \
  --band 0.06 --n-frames 60
echo "NRGBD CACHE DONE"
