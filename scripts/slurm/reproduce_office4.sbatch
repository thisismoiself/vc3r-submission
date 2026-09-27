#!/usr/bin/env bash
#SBATCH --job-name=repro_office4
#SBATCH --partition=NORMAL
#SBATCH --account=external
#SBATCH --qos=external
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/repro_office4_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/repro_office4_%j.err

# Reproduce figures/reproduction_report.pdf on the complete-target branch.
# From the repo root: sbatch scripts/slurm/reproduce_office4.sbatch
set -euo pipefail
export VC3R_REPO=/usr/prakt/s0017/vc3r
cd "$VC3R_REPO"
PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
export PATH="$(dirname "$PY"):$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1 DA3_LOG_LEVEL=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep the report's relative dataset argument, using the shared cluster data.
if [[ ! -e datasets/replica && ! -L datasets/replica ]]; then
  mkdir -p datasets
  ln -s /storage/group/cvpr/chwe/da3_nova3r/replica datasets/replica
fi
printf 'Job %s on %s; commit %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(git rev-parse HEAD)"
srun "$PY" -u scripts/stitch_office4.py \
  --ckpt checkpoints/final/nic/replica-nrgbd-7scenes-scannetpp.pt \
  --room office4 \
  --data-root datasets/replica \
  --complete-target \
  --stride 10 \
  --n-frames 8 \
  --num-queries 8192 \
  --fm-sampling euler \
  --fm-step-size 0.04 \
  --seed 42 \
  --out-tag repro_local
