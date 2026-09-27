#!/usr/bin/env bash
# Wait for the matched MSE-420 run to finish, evaluate its endpoint (*_best_last.pt) on held-out
# office4 with the same stitch recipe, and print the equal-data comparison:
# FM-420 vs MSE-420 (identical 420 windows) plus the full-data MSE-1225 baseline and the oracle.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKLAST=outputs/consecutive_windows/mse420_complete_best_last.pt
TRAINLOG=experiments/overfit_8frames/mse420_complete.log

echo "[wait] $(date) waiting for MSE-420 training to finish ..."
while ps -eo cmd | grep -q '[t]rain_online_var_hungarian'; do sleep 120; done
echo "[wait] $(date) training gone."; sleep 5
if [ ! -f "$CKLAST" ]; then echo "[FAIL] $CKLAST missing. Log tail:"; tail -25 "$TRAINLOG"; exit 1; fi

echo "[eval] $(date) stitching MSE-420 endpoint on office4"
$PY scripts/stitch_office4.py --ckpt "$CKLAST" --room office4 --complete-target \
    --fm-sampling midpoint --num-queries 50000 --out-tag mse420_office4 \
  > experiments/overfit_8frames/eval_mse420_office4.log 2>&1
echo "[eval] stitch exit=$?"

$PY - <<'PYEOF'
import json
def load(p):
    try: return json.load(open(p))
    except Exception: return None
runs = [
    ("FM-420 (final)",  "outputs/replica/stitch_office4_fm_only_office4_final/metrics.json"),
    ("MSE-420 (match)", "outputs/replica/stitch_office4_mse420_office4/metrics.json"),
    ("MSE-1225 (full)", "outputs/replica/stitch_office4_complete_office4/metrics.json"),
]
data = {n: load(p) for n, p in runs}
def line(name, d, key):
    if not d or key not in d: return f"  {name:<18} (missing)"
    x = d[key]; g=lambda k: x.get(k, float('nan'))
    return f"  {name:<18} chamfer={g('chamfer_m'):.4f}m  F@2={g('F@2cm'):.4f}  F@5={g('F@5cm'):.4f}"
print("\n===== office4 held-out: EQUAL-DATA (420) FM vs MSE, + full-data MSE + ORACLE =====")
print("WHOLE SCENE:")
for n,_ in runs: print(line(n, data[n], 'pred'))
print(line("ORACLE", data["MSE-1225 (full)"], 'oracle'))
print("FURNITURE REGION (detail metric):")
for n,_ in runs: print(line(n, data[n], 'pred_furniture'))
print(line("ORACLE", data["MSE-1225 (full)"], 'oracle_furniture'))
print("==================================================================================")
PYEOF
echo "[done] $(date)"
