#!/usr/bin/env bash
#SBATCH --job-name=fullrec_pca32
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_pca32_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_pca32_%j.out
# PCA-SUBSPACE TARGET A/B. Identical to the Replica-only full recipe (nf16, plain
# Hungarian MSE + velocity 0.3, 18k steps) EXCEPT the z* target is projected onto its
# top-32 PCA subspace (tail 96 dims -> pooled training mean). The truncate-and-decode
# sweep showed 32 dims reconstruct z* to ~0.7 cm (negligible ceiling loss) while 90% of
# the variance lives in only ~7. Tests whether removing the low-variance tail from the
# target helps the adapter focus on the decoder-sensitive dimensions.
#   baseline (full 128-dim target): outputs/consecutive_windows/fullrec_nf16_vel03_best.pt
#     -> office4 held out: whole 0.0567, F@5 0.707 | FURN F@2 0.371
# EXACTLY ONE VARIABLE CHANGES: the target subspace (--pca-project --pca-dims 32).
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --pca-project "$OUT/pca_basis_replica7.pt" --pca-dims 32 \
  --ckpt-out "$OUT/fullrec_nf16_vel03_pca32_best.pt" \
  > experiments/overfit_8frames/fullrec_nf16_vel03_pca32.log 2>&1
echo "=== [$(date)] TRAIN DONE exit=$? ==="

CK="$OUT/fullrec_nf16_vel03_pca32_best.pt"
if [ -f "$CK" ]; then
  echo "=== [$(date)] stitch eval office4 (held out) ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 \
      --fm-sampling midpoint --num-queries 50000 --out-tag fullrec_pca32_office4 \
      > experiments/overfit_8frames/eval_fullrec_pca32_office4_stitch.log 2>&1
  echo "=== [$(date)] EVAL exit=$? ==="
  $PY -c "import json; m=json.load(open('outputs/replica/stitch_office4_fullrec_pca32_office4/metrics.json')); p=m['pred']; pf=m['pred_furniture']; print(f\"PCA32: whole {p['chamfer_m']:.4f} wF@5 {p['F@5cm']:.3f} | FURN F@2 {pf['F@2cm']:.4f} F@5 {pf['F@5cm']:.4f}\"); print('baseline(128d): whole 0.0567 wF@5 0.707 | FURN F@2 0.371 F@5 0.667')"
else
  echo "=== [$(date)] NO CKPT — training failed ==="
fi
echo "=== [$(date)] ALL DONE ==="
