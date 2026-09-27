#!/usr/bin/env bash
#SBATCH --job-name=fullrec_da3geom
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_da3geom_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_da3geom_%j.out
# STEP 2 (deployable): full best recipe (nf16 1225 windows + plain + vel 0.3, 18k steps)
# + geometry conditioning on DA3 GT-free depth points (--geom-source da3). This is the
# deployable counterpart of the pts_norm upper bound (Step 1: FURN F@2 0.371->0.403).
# Depends on the da3pts cache job. a40 (48G/96G RAM); local 31G box OOM-kills the load.
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
  --geom-cond --geom-source da3 --geom-points 512 --geom-bands 16 \
  --ckpt-out "$OUT/fullrec_nf16_vel03_da3geom_best.pt" \
  > experiments/overfit_8frames/fullrec_da3geom.log 2>&1
echo "=== [$(date)] FULL RECIPE DA3-GEOM DONE exit=$? ==="

# --- auto stitch eval (same allocation) ---
CK="$OUT/fullrec_nf16_vel03_da3geom_best.pt"
if [ -f "$CK" ]; then
  echo "=== [$(date)] running stitch eval on $CK ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --fm-sampling midpoint --num-queries 50000 \
      --geom-points 512 --out-tag fullrec_nf16_vel03_da3geom_midpoint \
      > experiments/overfit_8frames/eval_fullrec_da3geom_stitch.log 2>&1
  echo "=== [$(date)] EVAL DONE exit=$? ==="
  $PY -c "import json; m=json.load(open('outputs/replica/stitch_office4_fullrec_nf16_vel03_da3geom_midpoint/metrics.json')); p=m['pred']; pf=m['pred_furniture']; print(f\"DA3-GEOM: whole {p['chamfer_m']:.4f} wF@5 {p['F@5cm']:.3f} FURN F@2 {pf['F@2cm']:.4f} F@5 {pf['F@5cm']:.4f}  | baseline 0.371  pts_norm-geom 0.403\")"
else
  echo "=== [$(date)] NO CKPT — training failed, skipping eval ==="
fi
