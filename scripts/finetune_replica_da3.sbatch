#!/bin/bash
#SBATCH --job-name=ft_da3
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/ft_da3_%j.log
# CORRECT Replica fine-tune: input = DA3 reconstruction (holes+noise), GT = clean mesh. Warm-start
# Stage-A best (94k), corrected 100%-coverage frontier loss, NO trajectory augmentation. Best-F
# watcher tracks the 3 held-out DA3 rooms (office3, office4=benchmark, room2).
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
CKPT=outputs/consecutive_windows/diffusion_replica_da3_ft.pt
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_replica_da3_ft_train --val-frac 0 \
  --init-from outputs/consecutive_windows/snapshots/stageA_v2_step94k_F81.pt \
  --ckpt $CKPT \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 8000 --batch-size 4 --lr 3e-5 --warmup 300 \
  --max-scenes 5 --workers 4 \
  --eval-every 100000 --ckpt-every 500 &
TRAIN_PID=$!
until [ -f "$CKPT" ]; do sleep 20; kill -0 $TRAIN_PID 2>/dev/null || { echo "train died"; exit 1; }; done
sleep 8
$PY scripts/progress_strip.py --watch --ckpt $CKPT \
  --cache-dir outputs/tsdf_cache_replica_da3_ft_holdout --val-frac 1.0 \
  --room-idx 0 --track-rooms 0 1 2 --steps 100 --milestone 500 --poll-min 3 \
  --best-ckpt outputs/consecutive_windows/snapshots/replica_da3_ft_BEST.pt \
  --best-json outputs/consecutive_windows/snapshots/replica_da3_ft_BEST.json \
  --history-dir outputs/consecutive_windows/snapshots/history_replica_da3_ft --history-keep 20 \
  --out-dir experiments/overfit_8frames/progress_strip_replica_da3_ft &
WATCH_PID=$!
wait $TRAIN_PID; kill $WATCH_PID 2>/dev/null; echo "FT DA3 DONE"
