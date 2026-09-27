#!/usr/bin/env bash
#SBATCH --job-name=stitch_gtfree
#SBATCH --partition=NORMAL
#SBATCH --account=external
#SBATCH --qos=external
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/stitch_gtfree/%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/stitch_gtfree/%j.err

# From the repo root: mkdir -p logs/stitch_gtfree
# sbatch scripts/slurm/stitch_gtfree.sbatch [additional stitch_office4.py arguments]
# Example small run: sbatch scripts/slurm/stitch_gtfree.sbatch --max-windows 1 --num-queries 256
# Later CLI arguments override the defaults below (e.g. --ckpt, --room, --out-tag).
set -euo pipefail
cd /usr/prakt/s0017/vc3r
PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
export PATH="$(dirname "$PY"):$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1 DA3_LOG_LEVEL=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf 'Job %s on %s at %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Match token conditioning to the adapter's training setup.
case "${POSE_SOURCE:-da3pred}" in
  da3pred) POSE_ARGS=(--da3pose-tokens) ;;
  gt) POSE_ARGS=() ;;
  *) echo "POSE_SOURCE must be da3pred or gt" >&2; exit 2 ;;
esac
srun "$PY" -u scripts/stitch_office4.py \
  --room office4 \
  --data-root /storage/group/cvpr/chwe/da3_nova3r/replica \
  --ckpt outputs/consecutive_windows/da3pose_nf16_vel03_complete_best.pt \
  --complete-target "${POSE_ARGS[@]}" \
  --n-frames 8 --stride 10 --fm-sampling midpoint --num-queries 50000 \
  --out-tag "demo_${SLURM_JOB_ID}" "$@"
