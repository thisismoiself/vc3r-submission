#!/bin/bash
#SBATCH --job-name=ft_replica
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/ft_replica_%j.log
# Stage-B FINE-TUNE on REPLICA (the real target). The Stage-A gap on Replica is a CONTENT/occlusion
# distribution mismatch between clean domains (3D-FRONT furniture rooms vs real Replica offices), NOT
# sensor noise -> fine-tuning on Replica directly attacks it. Warm-start from the frozen best Stage-A
# (step 78k, cross-room F=79.9); same v2 recipe (surf_w=1 + frontier-loss + cond-drop); gentle LR so
# the good init isn't washed out. TRAIN on 5 rooms (office0/1/2, room0/1); a GPU-sharing best-F
# watcher tracks the 3 HELD-OUT rooms (office3, office4=benchmark, room2) and promotes replica_ft_BEST
# by cross-room F (single-crop, 100 steps). Few scenes overfit fast -> the held-out F curve is the
# real signal and best-tracking captures the peak.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
CKPT=outputs/consecutive_windows/diffusion_replica_ft.pt

$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_replica_ft_train --val-frac 0 \
  --init-from outputs/consecutive_windows/snapshots/stageA_v2_step78k_F80.pt \
  --ckpt $CKPT \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 12000 --batch-size 4 --lr 3e-5 --warmup 300 \
  --max-scenes 40 --workers 4 \
  --eval-every 100000 --ckpt-every 500 &
TRAIN_PID=$!

until [ -f "$CKPT" ]; do sleep 20; kill -0 $TRAIN_PID 2>/dev/null || { echo "train died before first ckpt"; exit 1; }; done
sleep 8
$PY scripts/progress_strip.py --watch \
  --ckpt $CKPT \
  --cache-dir outputs/tsdf_cache_replica_ft_holdout --val-frac 1.0 \
  --room-idx 0 --track-rooms 0 1 2 \
  --steps 100 --milestone 500 --poll-min 3 \
  --best-ckpt outputs/consecutive_windows/snapshots/replica_ft_BEST.pt \
  --best-json outputs/consecutive_windows/snapshots/replica_ft_BEST.json \
  --history-dir outputs/consecutive_windows/snapshots/history_replica_ft \
  --history-keep 20 \
  --out-dir experiments/overfit_8frames/progress_strip_replica_ft &
WATCH_PID=$!

wait $TRAIN_PID
kill $WATCH_PID 2>/dev/null
echo "FT REPLICA DONE"
