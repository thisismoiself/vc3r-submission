#!/bin/bash
#SBATCH --job-name=v3_nrgbd
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_v3_nrgbd_%j.log
# v3 (F=84) whole-room stitch on the cached NeuralRGBD DA3 scenes -> completions for the 2nd-dataset test.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
for r in breakfast_room grey_white_room morning_apartment whiteroom; do
  echo "########## $r ##########"
  $PY scripts/stitch_room_diffusion.py \
    --npz outputs/tsdf_cache_nrgbd_da3_006/replica_$r.npz \
    --ckpt outputs/consecutive_windows/snapshots/stageA_v3_BEST.pt \
    --out-dir outputs/fullroom_stitch_v3_nrgbd --win 128 --stride 64 --steps 100
done
echo "V3 NRGBD DONE"
