#!/usr/bin/env bash
#SBATCH --job-name=fspan
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fspan_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fspan_%j.out
# Frame-count + span ablation: train plain-loss adapter on each distribution.
# Equal window count (--max-per-root 70) so distributions are the only difference.
# Frame axis @ span24-100: run1(nf4-10), nf8, nf16.  Span axis @ nf8: spanshort, nf8(anchor), spanlong.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=outputs/consecutive_windows; mkdir -p "$OUT"

echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

#         label        cache_dir
CACHES=(
  "run1_nf4_10:scripts/data/run1_span24_100_nf4_10_l13"
  "nf8:scripts/data/fc_nf8_span24_100_l13"
  "nf16:scripts/data/fc_nf16_span24_100_l13"
  "spanshort:scripts/data/fc_spanshort_16_36_nf8_l13"
  "spanlong:scripts/data/fc_spanlong_120_300_nf8_l13"
)

for entry in "${CACHES[@]}"; do
  LABEL=${entry%%:*}; C=${entry#*:}
  echo "=== [$(date)] TRAIN framespan=$LABEL  cache=$C ==="
  $PY scripts/train_online_var_hungarian.py \
    --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
    --val-rooms $C/office4 --max-per-root 70 --max-val-per-root 60 \
    --steps 12000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
    --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
    --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
    --loss-mode plain --ckpt-out "$OUT/fs_${LABEL}_best.pt" \
    > experiments/overfit_8frames/fs_${LABEL}.log 2>&1
  echo "=== [$(date)] DONE framespan=$LABEL exit=$? ==="
done
echo "=== [$(date)] FRAMESPAN TRAINING COMPLETE ==="
