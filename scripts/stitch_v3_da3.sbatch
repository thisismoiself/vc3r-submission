#!/bin/bash
#SBATCH --job-name=stitch_v3da3
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_v3da3_%j.log
# Evaluate the in-domain best v3 model (0.06m, F=84) on REAL DA3 input (0.06m caches), whole-room.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
$PY scripts/stitch_room_diffusion.py \
  --npz outputs/tsdf_cache_replica_da3_006/replica_office4.npz \
  --ckpt outputs/consecutive_windows/snapshots/stageA_v3_BEST.pt \
  --out-dir outputs/fullroom_stitch_v3_da3 --win 128 --stride 64 --steps 100
echo "V3 DA3 STITCH DONE"
