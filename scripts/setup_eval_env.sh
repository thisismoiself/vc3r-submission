#!/usr/bin/env bash
# Create or update the Miniconda environment used by the VC3R evaluator.
# Usage: scripts/setup_eval_env.sh [--prefix PATH] [--conda PATH]

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="$REPO_ROOT/.conda/envs/vc3r-eval"
CONDA_EXE=${CONDA_EXE:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { echo "--prefix requires a path" >&2; exit 2; }
            ENV_PREFIX=$2
            shift 2
            ;;
        --conda)
            [[ $# -ge 2 ]] || { echo "--conda requires a path" >&2; exit 2; }
            CONDA_EXE=$2
            shift 2
            ;;
        -h|--help)
            sed -n '2,3p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$CONDA_EXE" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_EXE=$(command -v conda)
    elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        CONDA_EXE="$HOME/miniconda3/bin/conda"
    else
        echo "Miniconda was not found. Pass --conda /path/to/miniconda3/bin/conda." >&2
        exit 1
    fi
fi

if [[ ! -x "$CONDA_EXE" ]]; then
    echo "Conda executable is not usable: $CONDA_EXE" >&2
    exit 1
fi

cd "$REPO_ROOT"
git submodule update --init --recursive

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
    echo "[env] updating $ENV_PREFIX"
    "$CONDA_EXE" env update --prefix "$ENV_PREFIX" --file environment-eval.yml --prune
else
    echo "[env] creating $ENV_PREFIX"
    "$CONDA_EXE" env create --prefix "$ENV_PREFIX" --file environment-eval.yml
fi

PYTHON="$ENV_PREFIX/bin/python"
echo "[env] python=$PYTHON"
"$PYTHON" --version

# Match the CUDA 12.1 stack used for the reported experiments. PyTorch3D is
# supplied by Conda above; installing the tested PyTorch wheel afterward
# reproduces the known-working vc3r-scripts environment on the cluster.
"$PYTHON" -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'
"$PYTHON" -m pip install --requirement requirements-eval.txt
"$PYTHON" -m pip install \
    'torch_cluster==1.6.3+pt25cu121' \
    --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

echo
echo "Environment ready. Activate it with:"
echo "  source \"$($CONDA_EXE info --base)/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_PREFIX\""
