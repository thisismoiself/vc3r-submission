#!/bin/bash
#SBATCH --job-name=v3da3_rest
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=14G
#SBATCH --time=05:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_v3da3_rest_%j.log
# v3 (F=84) whole-room stitch on the 5 remaining Replica DA3 rooms so all 8 can be benchmarked.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
for r in office0 office1 office3 room0 room2; do
  echo "########## $r ##########"
  $PY scripts/stitch_room_diffusion.py \
    --npz outputs/tsdf_cache_replica_da3_006/replica_$r.npz \
    --ckpt outputs/consecutive_windows/snapshots/stageA_v3_BEST.pt \
    --out-dir outputs/fullroom_stitch_v3_da3 --win 128 --stride 64 --steps 100
done
echo "REST DONE"
