#!/usr/bin/env bash
#SBATCH --job-name=adapter_infer
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/eval_niko/adapter_infer_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/eval_niko/adapter_infer_%j.out

set -euo pipefail

REPO=/usr/prakt/s0017/vc3r
PY=${VC3R_PYTHON:-$REPO/.conda/envs/vc3r-eval/bin/python}
DEFAULT_DATA_ROOT=/storage/group/cvpr/chwe/da3_nova3r/replica

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 CHECKPOINT [stitch_office4.py options...]" >&2
    echo "Example: sbatch $0 outputs/consecutive_windows/model.pt --room office4 --complete-target --max-windows 1 --num-queries 8192 --out-tag smoke" >&2
    exit 2
fi

CKPT=$1
shift

cd "$REPO"

if [[ ! -x "$PY" ]]; then
    echo "Python environment not found or not executable: $PY" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Adapter checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f checkpoints/nova3r/scene_ae/checkpoint-last.pth ]]; then
    echo "NOVA3R decoder checkpoint is missing: checkpoints/nova3r/scene_ae/checkpoint-last.pth" >&2
    exit 1
fi

export NOVA3R_DIR=nova3r
export PYTHONPATH="$REPO/nova3r/third_party:$REPO/nova3r:$REPO/da3/src:$REPO/scripts:$REPO/experiments/overfit_8frames${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT=${DATA_ROOT:-$DEFAULT_DATA_ROOT}

echo "=== [$(date)] adapter inference ==="
echo "host=$(hostname)"
echo "checkpoint=$CKPT"
echo "data_root=$DATA_ROOT"
echo "extra_args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$PY" -u scripts/stitch_office4.py \
    --ckpt "$CKPT" \
    --data-root "$DATA_ROOT" \
    "$@"

echo "=== [$(date)] inference completed ==="
