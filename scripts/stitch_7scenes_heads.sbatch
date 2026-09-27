#!/bin/bash
#SBATCH --job-name=adapt_7sc
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/stitch_7scenes/stitch_7scenes_%j.log
# DA3 -> NOVA3R complete-target adapter (Replica+NeuralRGBD ckpt) on REAL 7-Scenes 'heads'.
# Generalization test to a real RGB-D dataset. GT = Open3D TSDF fusion of 7-Scenes' own scan
# (incomplete, real) so metrics are accuracy-leaning. nf16 to match the adapter's training recipe.
# PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
# cd /usr/prakt/s0016/vc3r

cd /usr/prakt/s0017/vc3r
source /usr/prakt/s0017/miniconda3/etc/profile.d/conda.sh
conda activate vc3r-scripts

export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
python scripts/stitch_office4.py \
  --ckpt checkpoints/final/nic/flow-matching.pt \
  --complete-target \
  --data-root /storage/group/cvpr/chwe/da3_nova3r/7scenes/ --room heads \
  --n-frames 16 --stride 2 --max-windows 5 --num-queries 8192 \
  --out-tag 7scenes_heads
echo "7SCENES STITCH DONE"
