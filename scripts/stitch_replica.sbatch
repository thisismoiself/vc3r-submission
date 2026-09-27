#!/bin/bash
#SBATCH --job-name=stitch_repl
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_replica_%j.log
# Whole-room tiled diffusion completion on REPLICA office4 (the real deployment target), using the
# FROZEN best stageA_v2 checkpoint (step 78k, cross-room F=79.9). band=0.10 matches the model's
# training band (v2 recipe trained on the 0.10m cache). Proper SLURM job so it survives disconnects
# (the earlier local run died at window 6/50).
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
$PY scripts/stitch_room_diffusion.py \
  --npz outputs/tsdf_cache_replica/replica_office4.npz \
  --ckpt outputs/consecutive_windows/snapshots/stageA_v2_step78k_F80.pt \
  --out-dir outputs/fullroom_stitch_replica \
  --win 128 --stride 64 --steps 100
echo "STITCH REPLICA DONE"
