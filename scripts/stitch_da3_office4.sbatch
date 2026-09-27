#!/bin/bash
#SBATCH --job-name=stitch_da3
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_da3_office4_%j.log
# THE REAL DEPLOYMENT TEST: fill office4's DA3-reconstruction holes (partial = DA3, GT = clean mesh),
# whole-room, 100 steps. Stage-A best vs Replica-fine-tuned best.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
NPZ=outputs/tsdf_cache_replica_da3/replica_office4.npz
declare -A C=( [stageA]=outputs/consecutive_windows/snapshots/stageA_v2_step94k_F81.pt
               [ft]=outputs/consecutive_windows/snapshots/replica_ft_BEST.pt )
for tag in stageA ft; do
  echo "########## DA3-STITCH $tag ##########"
  $PY scripts/stitch_room_diffusion.py --npz $NPZ --ckpt ${C[$tag]} \
      --out-dir outputs/fullroom_stitch_da3_$tag --win 128 --stride 64 --steps 100
done
echo "DA3 STITCH DONE"
