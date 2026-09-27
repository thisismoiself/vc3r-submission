#!/usr/bin/env bash
# Run the Replica Office4 evaluator without a cluster scheduler.
# Usage: scripts/reproduce_office4.sh ADAPTER_CHECKPOINT REPLICA_ROOT [options...]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 ADAPTER_CHECKPOINT REPLICA_ROOT [stitch_office4.py options...]" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${VC3R_PYTHON:-$REPO/.conda/envs/vc3r-eval/bin/python}
CKPT=$1
DATA_ROOT=$2
shift 2

cd "$REPO"
[[ -x "$PY" ]] || { echo "Evaluation environment missing: $PY" >&2; exit 1; }

echo "commit=$(git rev-parse HEAD)"
echo "python=$PY data_root=$DATA_ROOT checkpoint=$CKPT"

exec "$PY" -u scripts/stitch_office4.py \
    --ckpt "$CKPT" \
    --data-root "$DATA_ROOT" \
    "$@"
