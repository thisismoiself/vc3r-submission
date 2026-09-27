#!/usr/bin/env bash
#SBATCH --job-name=fullrec_complete
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_complete_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fullrec_complete_%j.out
# COMPLETE-TARGET A/B. Step 1: re-cache z* on the COMPLETE (amodal) frustum pool matching
# NOVA3R's src_complete training (reusing existing da3_tokens via symlink -> only z* re-encoded).
# Step 2: retrain the adapter with the byte-identical Replica-only full recipe (nf16, plain
# Hungarian MSE + velocity 0.3, 18k steps) on the complete targets. EXACTLY ONE variable vs
# the visible baseline (fullrec_nf16_vel03_best.pt): the target is complete, not visible.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13_complete
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] STEP 1: re-cache complete targets (7 train rooms + office4) ==="
$PY -u scripts/recache_complete_target.py \
    > experiments/overfit_8frames/recache_complete.log 2>&1
echo "=== [$(date)] RECACHE exit=$? ==="
tail -3 experiments/overfit_8frames/recache_complete.log

if ! grep -q "RECACHE DONE" experiments/overfit_8frames/recache_complete.log; then
  echo "RECACHE FAILED — aborting before train"; exit 1
fi
echo "windows recached: $(ls -d $C/office0/*/ $C/office1/*/ $C/office2/*/ $C/office3/*/ $C/room0/*/ $C/room1/*/ $C/room2/*/ 2>/dev/null | wc -l) train + $(ls -d $C/office4/*/ 2>/dev/null | wc -l) val"

echo "=== [$(date)] STEP 2: train adapter on complete targets ==="
$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --ckpt-out "$OUT/fullrec_nf16_vel03_complete_best.pt" \
  > experiments/overfit_8frames/fullrec_nf16_vel03_complete.log 2>&1
echo "=== [$(date)] TRAIN exit=$? ==="
tail -3 experiments/overfit_8frames/fullrec_nf16_vel03_complete.log
echo "=== [$(date)] ALL DONE (eval: complete-stitch handled separately) ==="
