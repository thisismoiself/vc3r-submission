#!/bin/bash
#SBATCH --job-name=diff_da3sim
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_da3sim_%j.log
# DA3-SIM Stage-A: same recipe as v3 (0.06m v2 data, surf_w=1 + corrected frontier loss + CFG) but
# with --da3-sim, so the clean 3D-FRONT partials are corrupted to DA3-like (jitter+surface holes)
# on the fly -> teaches DA3-hole-filling. mem/scenes reduced to co-fit with v3 under the 128G cap;
# per-epoch corruption compensates for fewer loaded rooms. Eval on 0.06m DA3 caches (built separately).
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
CKPT=outputs/consecutive_windows/diffusion_stageA_da3sim.pt
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_front3d_v2 --band 0.06 --da3-sim \
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
  --best-ckpt outputs/consecutive_windows/snapshots/stageA_da3sim_BEST.pt \
  --best-json outputs/consecutive_windows/snapshots/stageA_da3sim_BEST.json \
  --history-dir outputs/consecutive_windows/snapshots/history_da3sim --history-keep 20 \
  --out-dir experiments/overfit_8frames/progress_strip_da3sim &
WATCH_PID=$!
wait $TRAIN_PID; kill $WATCH_PID 2>/dev/null; echo "DA3SIM DONE"
