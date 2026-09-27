#!/usr/bin/env bash
#SBATCH --job-name=complete_bestcombo
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/complete_bestcombo_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/complete_bestcombo_%j.out
# BEST-COMBO on the complete/amodal target: stack the confirmed detail lever (PCA-32 subspace,
# basis re-derived on COMPLETE tokens) onto plain Hungarian MSE + velocity 0.3. Goal: recover
# the accuracy dip of the plain complete-target adapter while keeping the completeness gain.
#   plain complete adapter: PRED comp 3.62 acc 6.75 cham 5.19 | oracle cham 2.71
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13_complete
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --pca-project "$OUT/pca_basis_replica7_complete.pt" --pca-dims 32 \
  --ckpt-out "$OUT/complete_bestcombo_pca32_best.pt" \
  > experiments/overfit_8frames/complete_bestcombo.log 2>&1
echo "=== [$(date)] TRAIN exit=$? ==="

CK="$OUT/complete_bestcombo_pca32_best.pt"
if [ -f "$CK" ]; then
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 --complete-target \
      --fm-sampling midpoint --num-queries 50000 --out-tag complete_bestcombo \
      > experiments/overfit_8frames/eval_complete_bestcombo_stitch.log 2>&1
  $PY - <<'PYEOF'
import json
b=json.load(open('outputs/replica/stitch_office4_complete_bestcombo/metrics.json'))['pred']
print(f"BESTCOMBO PRED: comp {b['completeness_m']*100:.2f} acc {b['accuracy_m']*100:.2f} cham {b['chamfer_m']*100:.2f} F@5 {b['F@5cm']:.3f} | FURN F@2 {json.load(open('outputs/replica/stitch_office4_complete_bestcombo/metrics.json'))['pred_furniture']['F@2cm']:.4f}")
print("plain complete: comp 3.62 acc 6.75 cham 5.19 F@5 0.672")
PYEOF
fi
echo "=== [$(date)] ALL DONE ==="
