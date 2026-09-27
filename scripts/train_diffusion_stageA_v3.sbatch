#!/bin/bash
#SBATCH --job-name=diff_sA_v3
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_stageA_v3_%j.log
# Combined Stage-A run: best loss recipe (surf_w=1 + CORRECTED 100%-coverage frontier loss + CFG)
# on the NEW v2 data (band 0.06m, dense-sweep trajectories, ~10.4k rooms). From scratch. mem=90G so
# it co-fits with the DA3 fine-tune under the 128G/20cpu cap. Best-F watcher on a 0.06m held-out set.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
CKPT=outputs/consecutive_windows/diffusion_stageA_v3.pt
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_front3d_v2 --band 0.06 \
  --ckpt $CKPT \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 200000 --batch-size 4 --lr 1e-4 --warmup 5000 \
  --max-scenes 1200 --workers 8 \
  --eval-every 5000 --ckpt-every 2000 &
TRAIN_PID=$!
until [ -f "$CKPT" ]; do sleep 30; kill -0 $TRAIN_PID 2>/dev/null || { echo "train died"; exit 1; }; done
sleep 10
$PY scripts/progress_strip.py --watch --ckpt $CKPT --band 0.06 \
  --cache-dir outputs/tsdf_cache_front3d_v2_holdout --val-frac 1.0 \
  --room-idx 0 --track-rooms 0 1 2 3 4 --steps 100 --milestone 2000 --poll-min 10 \
  --best-ckpt outputs/consecutive_windows/snapshots/stageA_v3_BEST.pt \
  --best-json outputs/consecutive_windows/snapshots/stageA_v3_BEST.json \
  --history-dir outputs/consecutive_windows/snapshots/history_v3 --history-keep 20 \
  --out-dir experiments/overfit_8frames/progress_strip_v3 &
WATCH_PID=$!
wait $TRAIN_PID; kill $WATCH_PID 2>/dev/null; echo "STAGE A v3 DONE"
