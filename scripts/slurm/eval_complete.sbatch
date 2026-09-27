#!/usr/bin/env bash
#SBATCH --job-name=eval_complete
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/eval_niko/eval_complete_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/eval_niko/eval_complete_%j.out
set -uo pipefail
cd /usr/prakt/s0017/vc3r
PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [$(date)] host=$(hostname) ==="
# complete-target adapter, complete-target stitch (amodal oracle + pts_norm)
$PY scripts/stitch_office4.py --ckpt outputs/consecutive_windows/fullrec_nf16_vel03_complete_best.pt \
    --room office4 --complete-target --fm-sampling midpoint --num-queries 50000 \
    --out-tag complete_office4 > experiments/overfit_8frames/eval_complete_office4_stitch.log 2>&1
echo "=== [$(date)] EVAL exit=$? ==="
$PY - <<'PYEOF'
import json
new=json.load(open('outputs/replica/stitch_office4_complete_office4/metrics.json'))
old=json.load(open('outputs/replica/stitch_office4_fullrec_nf16_vel03_midpoint/metrics.json'))
def row(tag,d): print(f"  {tag:26s} comp={d['completeness_m']*100:5.2f}cm  acc={d['accuracy_m']*100:5.2f}cm  cham={d['chamfer_m']*100:5.2f}cm  F@5={d['F@5cm']:.3f}")
print("\n===== WHOLE-SCENE office4: COMPLETE-target vs VISIBLE-baseline =====")
print("[PRED]"); row("visible target (old)",old['pred']); row("complete target (new)",new['pred'])
print("[ORACLE]"); row("visible target (old)",old['oracle']); row("complete target (new)",new['oracle'])
PYEOF
echo "=== [$(date)] DONE ==="
