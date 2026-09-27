#!/bin/bash
#SBATCH --job-name=stitch_room
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_room_%j.log
# Full-room completion by tiled diffusion (overlapping 128^3 windows + Hann cross-fade blend),
# on a band=0.10m held-out room (matches the model's training band). Uses stageA_v2_BEST.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
$PY scripts/stitch_room_diffusion.py \
  --npz outputs/tsdf_cache_front3d_holdout/01861393-3d82-4926-af4c-e2d29c6d0c0b_LivingRoom-31965.npz \
  --ckpt outputs/consecutive_windows/snapshots/stageA_v2_BEST.pt \
  --out-dir outputs/fullroom_stitch \
  --win 128 --stride 64 --steps 100
echo "STITCH DONE"
