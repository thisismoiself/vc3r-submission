#!/usr/bin/env bash
#SBATCH --job-name=pf_complete
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/pf_complete_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/pf_complete_%j.out
# MSE + FM finetune on COMPLETE tokens: Stage-2 point-space flow corrector on top of the
# frozen complete-target MSE adapter. Trains a corrector that transports decode(adapter(da3))
# toward the complete surface (gt = pts_norm), conditioned only on z_pred tokens (GT-free).
# Then A/B: office4 stitch MSE-only (existing stitch_office4_complete_office4) vs MSE+corrector.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13_complete
OUT=outputs/consecutive_windows
BASE=$OUT/fullrec_nf16_vel03_complete_best.pt
CORR=$OUT/point_flow_complete.pt
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== [$(date)] train point-flow corrector on complete tokens ==="
$PY scripts/train_point_flow.py --base-ckpt "$BASE" \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 175 --max-val-per-root 8 \
  --bank-queries 8192 --steps 8000 --batch-size 8 --n-couple 1024 --lr 3e-4 \
  --hidden 256 --depth 4 --flow-steps 6 --bank-dir outputs/point_flow_bank_complete \
  --ckpt-out "$CORR" \
  > experiments/overfit_8frames/point_flow_complete.log 2>&1
echo "=== [$(date)] TRAIN exit=$? ==="

if [ -f "$CORR" ]; then
  echo "=== [$(date)] stitch MSE+FM (corrector) ==="
  $PY scripts/stitch_office4.py --ckpt "$BASE" --room office4 --complete-target \
      --point-corrector "$CORR" --corrector-steps 6 \
      --fm-sampling midpoint --num-queries 50000 --out-tag complete_pf \
      > experiments/overfit_8frames/eval_complete_pf_stitch.log 2>&1
  $PY - <<'PYEOF'
import json
mse=json.load(open('outputs/replica/stitch_office4_complete_office4/metrics.json'))
fm =json.load(open('outputs/replica/stitch_office4_complete_pf/metrics.json'))
def row(t,m): p=m['pred'];pf=m['pred_furniture'];print(f"  {t:12s} cham={p['chamfer_m']*100:.2f} comp={p['completeness_m']*100:.2f} F@5={p['F@5cm']:.3f} | FURN F@2={pf['F@2cm']:.4f} F@5={pf['F@5cm']:.3f}")
print("\n===== complete tokens: MSE-only vs MSE+FM corrector (office4) =====")
row("MSE only",mse); row("MSE + FM",fm)
PYEOF
fi
echo "=== [$(date)] ALL DONE ==="
