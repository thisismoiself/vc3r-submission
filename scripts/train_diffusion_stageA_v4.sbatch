#!/bin/bash
#SBATCH --job-name=diff_sA_v4
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_stageA_v4_%j.log
# Stage-A v4 = v3 recipe (surf_w=1 + frontier-dilate 12 + CFG, band 0.06) DATA-SCALED via streaming
# ALL ~12.3k v2 rooms (vs v3's 1.2k). TRAIN-ONLY: the inline best-F watcher is removed because it
# time-shared our single GPU with training and deadlocked on CUDA allocs (twice). Best-F is tracked
# offline instead (eval saved checkpoints). Resumes from the step-14000 checkpoint if present.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_front3d_v2 --band 0.06 \
  --ckpt outputs/consecutive_windows/diffusion_stageA_v4.pt \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 200000 --batch-size 4 --lr 1e-4 --warmup 5000 \
  --pool-size 250 --refresh-every 400 --workers 2 --val-frac 0.002 \
  --eval-every 100000 --ckpt-every 2000
echo "STAGE A v4 DONE"
