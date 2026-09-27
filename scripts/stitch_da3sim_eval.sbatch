#!/bin/bash
#SBATCH --job-name=da3sim_eval
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/da3sim_eval_%j.log
# THE experiment's verdict: DA3-sim best (0.06m) on real DA3 input, vs v3's F=73. office4 + office2.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
for r in office4 office2; do
  echo "########## DA3-SIM on $r ##########"
  $PY scripts/stitch_room_diffusion.py \
    --npz outputs/tsdf_cache_replica_da3_006/replica_$r.npz \
    --ckpt outputs/consecutive_windows/snapshots/stageA_da3sim_BEST.pt \
    --out-dir outputs/fullroom_stitch_da3sim_eval --win 128 --stride 64 --steps 100
done
echo "DA3SIM EVAL DONE"
