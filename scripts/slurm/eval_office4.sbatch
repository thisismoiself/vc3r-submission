#!/usr/bin/env bash
#SBATCH --job-name=vc3r_office4
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=vc3r_office4_%j.out
#SBATCH --error=vc3r_office4_%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 ADAPTER_CHECKPOINT [stitch_office4.py options...]" >&2
    exit 2
fi

REPO=${VC3R_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
PY=${VC3R_PYTHON:-$REPO/.conda/envs/vc3r-eval/bin/python}
DATA_ROOT=${DATA_ROOT:-$REPO/datasets/replica}
CKPT=$1
shift

cd "$REPO"
[[ -x "$PY" ]] || { echo "Evaluation environment missing: $PY" >&2; exit 1; }
[[ -f "$CKPT" ]] || { echo "Adapter checkpoint missing: $CKPT" >&2; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
export DA3_LOG_LEVEL=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_JOB_ID:-none} host=$(hostname) commit=$(git rev-parse HEAD)"
echo "python=$PY data_root=$DATA_ROOT checkpoint=$CKPT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "$PY" -u scripts/stitch_office4.py \
    --ckpt "$CKPT" \
    --data-root "$DATA_ROOT" \
    "$@"
