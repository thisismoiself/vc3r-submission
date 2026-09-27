#!/usr/bin/env bash
# Wait for the pure-FM run to finish, then evaluate the step-4000 snapshot AND the final endpoint
# on held-out office4 (same stitch recipe as the MSE baseline), and print the 4k->final trajectory
# vs MSE vs oracle. Checkpoint stem is 'fm_only_complete_best', so snapshots are *_best_step{N}.pt
# and the endpoint is *_best_last.pt.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK4K=outputs/consecutive_windows/fm_only_complete_best_step4000.pt
CKLAST=outputs/consecutive_windows/fm_only_complete_best_last.pt
TRAINLOG=experiments/overfit_8frames/fm_only_complete.log

echo "[wait] $(date) waiting for training to finish (endpoint $CKLAST) ..."
while ps -eo cmd | grep -q '[t]rain_online_var_hungarian'; do sleep 120; done
echo "[wait] $(date) training process gone."
sleep 5

eval_one () {  # $1=ckpt  $2=tag
  local ck="$1" tag="$2"
  if [ ! -f "$ck" ]; then echo "[skip] $ck missing"; return; fi
  echo "[eval] $(date) $tag -> $ck"
  $PY scripts/stitch_office4.py --ckpt "$ck" --room office4 --complete-target \
      --fm-sampling midpoint --num-queries 50000 --out-tag "$tag" \
    > "experiments/overfit_8frames/eval_${tag}.log" 2>&1
  echo "[eval] $tag stitch exit=$?"
}
eval_one "$CK4K"   fm_only_office4_4k
eval_one "$CKLAST" fm_only_office4_final

$PY - <<'PYEOF'
import json
def load(p):
    try: return json.load(open(p))
    except Exception: return None
runs = [
    ("FM-only @4k",    "outputs/replica/stitch_office4_fm_only_office4_4k/metrics.json"),
    ("FM-only @final", "outputs/replica/stitch_office4_fm_only_office4_final/metrics.json"),
    ("MSE baseline",   "outputs/replica/stitch_office4_complete_office4/metrics.json"),
]
data = {n: load(p) for n, p in runs}
def line(name, d, key):
    if not d or key not in d: return f"  {name:<16} (missing)"
    x = d[key]; g=lambda k: x.get(k, float('nan'))
    return f"  {name:<16} chamfer={g('chamfer_m'):.4f}m  F@2={g('F@2cm'):.4f}  F@5={g('F@5cm'):.4f}"
print("\n============ office4 held-out: pure-FM trajectory vs MSE vs ORACLE ============")
print("WHOLE SCENE:")
for n,_ in runs: print(line(n, data[n], 'pred'))
print(line("ORACLE", data["MSE baseline"], 'oracle'))
print("FURNITURE REGION (detail metric):")
for n,_ in runs: print(line(n, data[n], 'pred_furniture'))
print(line("ORACLE", data["MSE baseline"], 'oracle_furniture'))
print("===============================================================================")
PYEOF
echo "[done] $(date)"
