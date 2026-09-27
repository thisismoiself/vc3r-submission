#!/usr/bin/env bash
#SBATCH --job-name=zcache
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/zcache_%A_%a.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/zcache_%A_%a.out
# Phase 1: cache the 4 frame-count / span distributions, one per array task on a dedicated GPU.
# Resumes by skipping already-cached windows. span fixed when varying frames; frames fixed when varying span.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib

#                  span_min span_max nf_min nf_max  out_root
CFG_0=(24  100  8  9   scripts/data/fc_nf8_span24_100_l13)        # frame axis: 8 frames (deployment)
CFG_1=(24  100  12 24  scripts/data/fc_nf16_span24_100_l13)       # frame axis: 12-24 frames (completeness)
CFG_2=(16  36   8  9   scripts/data/fc_spanshort_16_36_nf8_l13)   # span axis: short span (fine detail)
CFG_3=(120 300  8  9   scripts/data/fc_spanlong_120_300_nf8_l13)  # span axis: long span (coarse)

declare -n CFG="CFG_${SLURM_ARRAY_TASK_ID}"
SPAN_MIN=${CFG[0]}; SPAN_MAX=${CFG[1]}; NF_MIN=${CFG[2]}; NF_MAX=${CFG[3]}; OUT=${CFG[4]}

echo "=== [$(date)] host=$(hostname) span=[$SPAN_MIN,$SPAN_MAX] nf=[$NF_MIN,$NF_MAX] -> $OUT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/cache_online_hungarian_zstar_windows.py \
  --replica-root /usr/prakt/s0016/Replica --all-scenes --windows-per-room 70 \
  --span-min "$SPAN_MIN" --span-max "$SPAN_MAX" \
  --n-frames-min "$NF_MIN" --n-frames-max "$NF_MAX" \
  --n-seeds 30 --da3-layer 1 3 --da3-max-tokens 2048 \
  --image-height 392 --image-width 518 --flip \
  --out-root "$OUT"
echo "=== [$(date)] DONE $OUT exit=$? ==="
