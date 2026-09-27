#!/usr/bin/env bash
#SBATCH --job-name=nf16cache
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/nf16cache_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/nf16cache_%j.out
# Scale up the WINNING distribution (nf16: 12-24 frames, span 24-100) to 175/room.
# Resumes by skipping the 70/room already cached (seeded sampler -> first 70 identical).
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/cache_online_hungarian_zstar_windows.py \
  --replica-root /usr/prakt/s0016/Replica --all-scenes --windows-per-room 175 \
  --span-min 24 --span-max 100 --n-frames-min 12 --n-frames-max 24 \
  --n-seeds 30 --da3-layer 1 3 --da3-max-tokens 2048 \
  --image-height 392 --image-width 518 --flip \
  --out-root scripts/data/fc_nf16_span24_100_l13
echo "=== [$(date)] NF16 FULL CACHE DONE exit=$? ==="
