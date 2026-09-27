#!/bin/bash
#SBATCH --job-name=diff_sA_v5
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_stageA_v5_%j.log
# Stage-A v5 = the PROPER data-scaling experiment. v4 was flawed: refresh_every=400 made it overfit a
# slowly-drifting ~250-room cohort (each seen ~400x) -> F plateaued ~55-68, below v3's 84. v5 fixes the
# SAMPLING: fast refresh (every 40 samples) with pool 300 => residence ~3k steps, each room seen ~40x
# then rotated, cycling through all ~12k rooms => near-i.i.d. coverage = true data scaling. Same recipe
# otherwise (band 0.06, surf_w 1, frontier-dilate 12, CFG). From scratch. F logged from step 0 by an
# EXTERNAL periodic eval (no inline watcher -> no GPU-share deadlock).
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_front3d_v2 --band 0.06 \
  --ckpt outputs/consecutive_windows/diffusion_stageA_v5.pt \
  --surf-w 1.0 --frontier-dilate 12 --cond-drop 0.10 \
  --steps 200000 --batch-size 4 --lr 1e-4 --warmup 5000 \
  --pool-size 300 --refresh-every 40 --workers 2 --val-frac 0.002 \
  --eval-every 100000 --ckpt-every 2000
echo "STAGE A v5 DONE"
