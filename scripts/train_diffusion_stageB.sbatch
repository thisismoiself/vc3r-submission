#!/bin/bash
#SBATCH --job-name=diff_stageB
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/diff_stageB_%j.log
# Stage B: fine-tune the best Stage-A model on REAL ScanNet++ scans. Warm-start from stageA_BEST
# with a FRESH optimizer, the CORRECTED loss (surf_w 5->2 to curb over-prediction), a SEPARATE
# checkpoint file (never clobbers Stage A), and a larger val hold-out for real-data eval.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/train_diffusion.py \
  --cache-dir outputs/tsdf_cache_scannetpp \
  --init-from outputs/consecutive_windows/snapshots/stageA_BEST.pt \
  --ckpt outputs/consecutive_windows/diffusion_stageB.pt \
  --surf-w 2.0 --lr 5e-5 --warmup 1000 --steps 60000 \
  --max-scenes 600 --val-frac 0.05 --workers 8 \
  --eval-every 2000 --ckpt-every 2000
echo "STAGE B DONE"
