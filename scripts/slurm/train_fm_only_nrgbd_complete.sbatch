#!/usr/bin/env bash
#SBATCH --job-name=fm_nrgbd_complete
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fm_nrgbd_complete_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fm_nrgbd_complete_%j.out
# PURE FLOW-MATCHING (velocity distillation, --mse-weight 0 --vel-loss-weight 1.0) on the FULL
# combined set: 7 Replica-complete + 8 NRGBD-complete rooms (2084 windows), holding out office4
# (Replica) AND breakfast_room (NRGBD) for validation. Same recipe/schedule as the combined MSE run
# (train_nrgbd_complete_holdout.sbatch, which fit 2084 windows non-lazy in 120G) -- ONLY the loss
# differs. Complete-target caches already built, so no re-cache step. Periodic snapshots every 1000
# steps; endpoint (*_best_last.pt) evaluated on BOTH held-out scenes (token-MSE 'best' is misleading
# for pure FM).
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/usr/prakt/s0016/vc3r/.hf_cache
RC=scripts/data/fc_nf16_span24_100_l13_complete
NC=scripts/data/nrgbd_nf16_span24_100_l13_complete
OUT=outputs/consecutive_windows; mkdir -p "$OUT" scripts/slurm/logs
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/train_online_var_hungarian.py \
  --rooms $RC/office0 $RC/office1 $RC/office2 $RC/office3 $RC/room0 $RC/room1 $RC/room2 \
          $NC/complete_kitchen $NC/green_room $NC/grey_white_room $NC/kitchen \
          $NC/morning_apartment $NC/staircase $NC/thin_geometry $NC/whiteroom \
  --val-rooms $RC/office4 $NC/breakfast_room --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 --seed 42 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 --ckpt-every 1000 \
  --loss-mode plain --mse-weight 0 --vel-loss-weight 1.0 --vel-points 1024 \
  --ckpt-out "$OUT/fm_only_nrgbd_complete_best.pt" \
  > experiments/overfit_8frames/fm_only_nrgbd_complete.log 2>&1
echo "=== [$(date)] TRAIN exit=$? ==="

CK="$OUT/fm_only_nrgbd_complete_best_last.pt"
if [ -f "$CK" ]; then
  echo "=== [$(date)] EVAL office4 (Replica held-out) ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room office4 --complete-target \
      --fm-sampling midpoint --num-queries 50000 --out-tag fm_nrgbd_office4 \
      > experiments/overfit_8frames/eval_fm_nrgbd_office4.log 2>&1
  echo "=== [$(date)] EVAL breakfast_room (NRGBD held-out) ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --room breakfast_room --data-root /usr/prakt/s0016/NeuralRGBD \
      --complete-target --fm-sampling midpoint --num-queries 50000 --out-tag fm_nrgbd_bfr \
      > experiments/overfit_8frames/eval_fm_nrgbd_bfr.log 2>&1
  echo "=== [$(date)] EVALS DONE ==="
else
  echo "=== [$(date)] no *_best_last.pt -- training likely failed ==="; tail -20 experiments/overfit_8frames/fm_only_nrgbd_complete.log
fi
echo "=== [$(date)] ALL DONE ==="
