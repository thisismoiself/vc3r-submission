#!/bin/bash
#SBATCH --job-name=diff_sA_v2
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_stageA_v2_%j.log
# Stage-A v2: the precision-focused recipe on the EXISTING 0.10m cache (isolates the train/inference
# levers from the data levers, which land later via the 0.06m recache):
#   surf_w=1 (uniform, paper)  +  frontier-restricted loss (12 vox)  +  cond-drop 0.10 (CFG)
# A GPU-sharing best-F watcher tracks a 5-room HELD-OUT set and promotes stageA_v2_BEST by a
# precision-aware F (100 sampling steps), so an oscillation peak is never lost again.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
CKPT=outputs/consecutive_windows/diffusion_stageA_v2.pt

# --- training (background) ---
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_front3d \
  --ckpt $CKPT \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 200000 --batch-size 4 --lr 1e-4 --warmup 5000 \
  --max-scenes 1600 --workers 8 \
  --eval-every 5000 --ckpt-every 2000 &
TRAIN_PID=$!

# --- best-F watcher: wait for the first checkpoint, then poll & promote (shares the GPU) ---
until [ -f "$CKPT" ]; do sleep 30; kill -0 $TRAIN_PID 2>/dev/null || { echo "train died before first ckpt"; exit 1; }; done
sleep 10
$PY scripts/progress_strip.py --watch \
  --ckpt $CKPT \
  --cache-dir outputs/tsdf_cache_front3d_holdout --val-frac 1.0 \
  --room-idx 0 --track-rooms 0 1 2 3 4 \
  --steps 100 --milestone 2000 --poll-min 10 \
  --best-ckpt outputs/consecutive_windows/snapshots/stageA_v2_BEST.pt \
  --best-json outputs/consecutive_windows/snapshots/stageA_v2_BEST.json \
  --history-dir outputs/consecutive_windows/snapshots/history_v2 \
  --history-keep 20 \
  --out-dir experiments/overfit_8frames/progress_strip_v2 &
WATCH_PID=$!

wait $TRAIN_PID
kill $WATCH_PID 2>/dev/null
echo "STAGE A v2 DONE"
