#!/bin/bash
#SBATCH --job-name=v5peak
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/stitch_v5peak_%j.log
# Re-evaluate the whole-room Replica benchmark on the v5 F=77.5 PEAK checkpoint (periodicBEST, step 130000).
# Same stitch settings as the prior v3/v5 min-area benchmark (win128/stride64/steps100) for apples-to-apples.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
CKPT=outputs/consecutive_windows/snapshots/stageA_v5_periodicBEST.pt
OUT=outputs/fullroom_stitch_v5peak_da3
for r in office2 office4 room1; do
  echo "########## stitch $r ##########"
  $PY scripts/stitch_room_diffusion.py \
    --npz outputs/tsdf_cache_replica_da3_006/replica_$r.npz \
    --ckpt $CKPT --out-dir $OUT --win 128 --stride 64 --steps 100
done
echo "########## eval (min-area OBB) ##########"
$PY scripts/eval_replica_v3.py --rooms office2 office4 room1 \
  --stitch-dir $OUT --csv outputs/v5peak_replica_benchmark_minarea.csv
echo "V5PEAK DONE"
