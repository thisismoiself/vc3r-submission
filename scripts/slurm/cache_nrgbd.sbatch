#!/usr/bin/env bash
#SBATCH --job-name=zcache_nrgbd
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/zcache_nrgbd_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/zcache_nrgbd_%j.out
# Cache NeuralRGBD (9 Blender scenes) with the IDENTICAL nf16 recipe used for Replica
# (scripts/slurm/cache_all.sbatch: span [24,100], nf [12,24], n-seeds 30, layers 1+3,
# 2048 tokens, 392x518, flip) so z_star targets stay in one distribution and the new
# windows are drop-in for the trainer.
#
# Window budget = CONSTANT DENSITY with Replica (175 windows / 2000 frames = 1 per 11.43),
# NOT a flat 175/scene: spans are 24-100 frames, so 175 windows over thin_geometry's 395
# frames would be near-duplicate windows (repetition, not data). Totals ~961 windows.
# Every scene lands under 175, so the trainer's --max-per-root 175 passes them through.
#
# Distinct --sample-seed per scene: the sampler seeds its RNGs off room_index, which is 0
# in every single-scene invocation, so without this all scenes share a window pattern.
# Resumes by skipping already-cached windows.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
ROOT=/usr/prakt/s0016/NeuralRGBD
OUT=scripts/data/nrgbd_nf16_span24_100_l13

COMMON="--replica-root $ROOT --n-seeds 30 --da3-layer 1 3 --da3-max-tokens 2048 \
  --image-height 392 --image-width 518 --flip \
  --span-min 24 --span-max 100 --n-frames-min 12 --n-frames-max 24 --out-root $OUT"

echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# scene:windows(=round(frames/11.43)):seed
for entry in \
  breakfast_room:102:11 \
  complete_kitchen:106:12 \
  green_room:126:13 \
  grey_white_room:131:14 \
  kitchen:133:15 \
  morning_apartment:80:16 \
  staircase:101:17 \
  thin_geometry:35:18 \
  whiteroom:147:19 ; do
  scene="${entry%%:*}"; rest="${entry#*:}"; nwin="${rest%%:*}"; seed="${rest##*:}"
  echo "=== [$(date)] CACHE $scene  windows=$nwin seed=$seed ==="
  $PY scripts/cache_online_hungarian_zstar_windows.py $COMMON \
    --rooms "$scene" --windows-per-room "$nwin" --sample-seed "$seed"
  echo "=== [$(date)] DONE $scene exit=$? ==="
done

echo "=== [$(date)] ALL NRGBD CACHING DONE ==="
for d in "$OUT"/*/; do echo "$(basename "$d"): $(ls "$d" | wc -l) windows"; done
echo "TOTAL: $(ls -d "$OUT"/*/*/ 2>/dev/null | wc -l) windows"
