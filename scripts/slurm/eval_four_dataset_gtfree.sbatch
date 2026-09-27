#!/usr/bin/env bash
#SBATCH --job-name=eval_4data_gtfree
#SBATCH --partition=NORMAL
#SBATCH --account=external
#SBATCH --qos=external
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/stitch_gtfree/four_dataset_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/stitch_gtfree/four_dataset_%j.err

# Evaluate the four-dataset adapter with DA3-predicted pose tokens.
# GT poses remain in use for target construction, metric scale, world placement, and metrics.
set -euo pipefail
cd /usr/prakt/s0017/vc3r

PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
CKPT=checkpoints/final/nic/replica-nrgbd-7scenes-scannetpp.pt
DATA=/storage/group/cvpr/chwe/da3_nova3r

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1 DA3_LOG_LEVEL=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
  --ckpt "$CKPT"
  --complete-target
  --da3pose-tokens
  --n-frames 8
  --stride 10
  --fm-sampling midpoint
  --num-queries 50000
)

echo "=== office4 (Replica) ==="
srun "$PY" -u scripts/stitch_office4.py "${COMMON[@]}" \
  --data-root "$DATA/replica" --room office4 \
  --out-tag four_dataset_gtfree

echo "=== staircase (NeuralRGBD) ==="
srun "$PY" -u scripts/stitch_office4.py "${COMMON[@]}" \
  --data-root "$DATA/NeuralRGBD" --room staircase \
  --out-tag four_dataset_gtfree

echo "=== heads sequence 1 (7-Scenes) ==="
srun "$PY" -u scripts/stitch_office4.py "${COMMON[@]}" \
  --data-root "$DATA/7scenes" --room heads \
  --sequence-dir "$DATA/7scenes/heads/seq-01" \
  --gt-mesh "$DATA/7scenes/heads_mesh.ply" \
  --out-tag four_dataset_gtfree_seq01

echo "=== all evaluations complete ==="
