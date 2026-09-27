#!/usr/bin/env bash
#SBATCH --job-name=zcache
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/zcache_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/zcache_%j.out
# Phase 1: cache 4 frame-count/span distributions sequentially on one dedicated GPU.
# Order: nf8 (anchor, both axes) -> spanshort (detail, goal-relevant) -> nf16 -> spanlong.
# Resumes by skipping already-cached windows.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
COMMON="--replica-root /usr/prakt/s0016/Replica --all-scenes --windows-per-room 70 \
  --n-seeds 30 --da3-layer 1 3 --da3-max-tokens 2048 \
  --image-height 392 --image-width 518 --flip"

echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_cache () {  # span_min span_max nf_min nf_max out_root label
  echo "=== [$(date)] CACHE $6  span=[$1,$2] nf=[$3,$4] ==="
  $PY scripts/cache_online_hungarian_zstar_windows.py $COMMON \
    --span-min "$1" --span-max "$2" --n-frames-min "$3" --n-frames-max "$4" \
    --out-root "$5"
  echo "=== [$(date)] DONE $6 exit=$? ==="
}

run_cache 24  100  8  9   scripts/data/fc_nf8_span24_100_l13        nf8
run_cache 16  36   8  9   scripts/data/fc_spanshort_16_36_nf8_l13   spanshort
run_cache 24  100  12 24  scripts/data/fc_nf16_span24_100_l13       nf16
run_cache 120 300  8  9   scripts/data/fc_spanlong_120_300_nf8_l13  spanlong
echo "=== [$(date)] ALL CACHING DONE ==="
