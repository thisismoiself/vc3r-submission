#!/usr/bin/env bash
#SBATCH --job-name=nrgbd_complete
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/nrgbd_complete_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/nrgbd_complete_%j.out
# COMPLETE-target + cross-dataset. Step 1: re-cache NRGBD z* on the complete frustum pool
# (reuse da3_tokens). Step 2: train on 7 Replica-complete + 8 NRGBD-complete, holding out
# office4 (Replica) AND breakfast_room (NRGBD). Step 3: eval both held-out scenes.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RC=scripts/data/fc_nf16_span24_100_l13_complete
NC=scripts/data/nrgbd_nf16_span24_100_l13_complete
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] STEP 1: re-cache NRGBD complete targets (9 scenes) ==="
$PY -u scripts/recache_complete_target.py \
    --rooms breakfast_room complete_kitchen green_room grey_white_room kitchen morning_apartment staircase thin_geometry whiteroom \
    --data-root /usr/prakt/s0016/NeuralRGBD \
    --src scripts/data/nrgbd_nf16_span24_100_l13 --dst "$NC" \
    > experiments/overfit_8frames/recache_nrgbd_complete.log 2>&1
echo "=== [$(date)] RECACHE exit=$? ==="; tail -2 experiments/overfit_8frames/recache_nrgbd_complete.log
if ! grep -q "RECACHE DONE" experiments/overfit_8frames/recache_nrgbd_complete.log; then echo "RECACHE FAILED"; exit 1; fi

echo "=== [$(date)] STEP 2: train (Replica-7-complete + NRGBD-8-complete; hold out office4 + breakfast_room) ==="
$PY scripts/train_online_var_hungarian.py \
  --rooms $RC/office0 $RC/office1 $RC/office2 $RC/office3 $RC/room0 $RC/room1 $RC/room2 \
          $NC/complete_kitchen $NC/green_room $NC/grey_white_room $NC/kitchen \
          $NC/morning_apartment $NC/staircase $NC/thin_geometry $NC/whiteroom \
  --val-rooms $RC/office4 $NC/breakfast_room --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --ckpt-out "$OUT/fullrec_nrgbd_complete_holdout_bfr_best.pt" \
  > experiments/overfit_8frames/fullrec_nrgbd_complete.log 2>&1
echo "=== [$(date)] TRAIN exit=$? ==="

CK="$OUT/fullrec_nrgbd_complete_holdout_bfr_best.pt"
if [ -f "$CK" ]; then
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 --complete-target \
      --fm-sampling midpoint --num-queries 50000 --out-tag nrgbd_complete_office4 \
      > experiments/overfit_8frames/eval_nrgbd_complete_office4.log 2>&1
  $PY scripts/stitch_office4.py --ckpt "$CK" --room breakfast_room --data-root /usr/prakt/s0016/NeuralRGBD \
      --complete-target --fm-sampling midpoint --num-queries 50000 --out-tag nrgbd_complete_bfr \
      > experiments/overfit_8frames/eval_nrgbd_complete_bfr.log 2>&1
  echo "=== [$(date)] EVALS DONE ==="
fi
echo "=== [$(date)] ALL DONE ==="
