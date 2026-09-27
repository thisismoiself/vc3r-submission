#!/usr/bin/env bash
#SBATCH --job-name=eval_spp_gtfree
#SBATCH --partition=NORMAL
#SBATCH --account=external
#SBATCH --qos=external
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/stitch_scannetpp/%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/stitch_scannetpp/%j.err

# Evaluates the two selected ScanNet++ scenes sequentially in one allocation.
# Usage: sbatch scripts/slurm/eval_scannetpp_gtfree.sbatch
set -euo pipefail
cd /usr/prakt/s0017/vc3r

PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
MIRROR=/storage/group/dataset_mirrors/01_incoming/scannetpp
SCENE_IDS=(25f3b7a318 27dd4da69e)
STAGE_BASE=$(mktemp -d /tmp/scannetpp_stitch_${SLURM_JOB_ID}_XXXXXX)
cleanup() { rm -rf -- "$STAGE_BASE"; }
trap cleanup EXIT
CKPT=checkpoints/final/nic/replica-nrgbd-7scenes-scannetpp.pt

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1 DA3_LOG_LEVEL=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SCENE_ID in "${SCENE_IDS[@]}"; do
  RAW_SCENE="$MIRROR/data/$SCENE_ID"
  STAGE_ROOT="$STAGE_BASE/$SCENE_ID"

  echo "=== stage ScanNet++ scene $SCENE_ID from mirror ==="
  srun "$PY" -u scripts/prepare_scannetpp_stitch.py \
    --mirror "$MIRROR" --scene "$SCENE_ID" --out "$STAGE_ROOT" \
    --n-frames 1000 --width 960

  echo "=== evaluate $SCENE_ID with DA3-predicted pose tokens ==="
  srun "$PY" -u scripts/stitch_office4.py \
    --ckpt "$CKPT" \
    --data-root "$STAGE_ROOT" --room "$SCENE_ID" \
    --gt-mesh "$RAW_SCENE/scans/mesh_aligned_0.05.ply" \
    --complete-target --da3pose-tokens \
    --n-frames 8 --stride 10 --fm-sampling midpoint --num-queries 50000 \
    --out-tag four_dataset_gtfree_scannetpp

  echo "=== ScanNet++ evaluation complete: $SCENE_ID ==="
done
