#!/usr/bin/env bash
#SBATCH --job-name=fr_nrgbd_hobfr
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fr_nrgbd_hobfr_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fr_nrgbd_hobfr_%j.out
#
# CROSS-DATASET GENERALISATION experiment. Same proven full recipe (plain Hungarian MSE +
# velocity 0.3, nf16 span24-100), trained on Replica-6 + NeuralRGBD-8 with TWO scenes held
# out entirely for validation:
#   * office4       (Replica)   -- unchanged held-out scene, comparable to all prior numbers
#   * breakfast_room (NeuralRGBD) -- NEW held-out NRGBD scene (table+chairs; furniture story)
# Differs from train_fullrecipe_nrgbd.sbatch by EXACTLY the split: breakfast_room moves from
# train to val. Architecture / loss / schedule / steps / batch / LR / dropout all identical,
# so pred-vs-oracle gaps are directly comparable across the two held-out scenes.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13
N=scripts/data/nrgbd_nf16_span24_100_l13
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
          $N/complete_kitchen $N/green_room $N/grey_white_room \
          $N/kitchen $N/morning_apartment $N/staircase $N/thin_geometry $N/whiteroom \
  --val-rooms $C/office4 $N/breakfast_room --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --ckpt-out "$OUT/fullrec_nrgbd_holdout_bfr_best.pt" \
  > experiments/overfit_8frames/fullrec_nrgbd_holdout_bfr.log 2>&1
echo "=== [$(date)] TRAIN DONE exit=$? ==="

CK="$OUT/fullrec_nrgbd_holdout_bfr_best.pt"
if [ -f "$CK" ]; then
  # held-out office4 (Replica)
  echo "=== [$(date)] stitch eval office4 (Replica, held out) ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 \
      --fm-sampling midpoint --num-queries 50000 --out-tag nrgbd_hobfr_office4 \
      > experiments/overfit_8frames/eval_hobfr_office4_stitch.log 2>&1
  echo "=== [$(date)] office4 EVAL exit=$? ==="
  # held-out breakfast_room (NeuralRGBD)
  echo "=== [$(date)] stitch eval breakfast_room (NeuralRGBD, held out) ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room breakfast_room \
      --data-root /usr/prakt/s0016/NeuralRGBD \
      --fm-sampling midpoint --num-queries 50000 --out-tag nrgbd_hobfr_bfr \
      > experiments/overfit_8frames/eval_hobfr_bfr_stitch.log 2>&1
  echo "=== [$(date)] breakfast_room EVAL exit=$? ==="
else
  echo "=== [$(date)] NO CKPT — training failed ==="
fi
echo "=== [$(date)] ALL DONE ==="
