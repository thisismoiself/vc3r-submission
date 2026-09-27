#!/bin/bash
#SBATCH --job-name=v5_da3
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_v5_da3_%j.log
# v5 current-best (data-scaled, F=76.3) whole-room stitch on Replica DA3 caches -> compare to v3 (F=84).
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
for r in office2 office4 room1; do
  echo "########## $r ##########"
  $PY scripts/stitch_room_diffusion.py \
    --npz outputs/tsdf_cache_replica_da3_006/replica_$r.npz \
    --ckpt outputs/consecutive_windows/snapshots/stageA_v5_localBEST.pt \
    --out-dir outputs/fullroom_stitch_v5_da3 --win 128 --stride 64 --steps 100
done
echo "V5 DA3 DONE"
