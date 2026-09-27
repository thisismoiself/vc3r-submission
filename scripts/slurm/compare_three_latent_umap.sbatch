#!/usr/bin/env bash
#SBATCH --job-name=three_umap
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/eval_niko/three_umap_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/eval_niko/three_umap_%j.out

set -euo pipefail
cd /usr/prakt/s0017/vc3r

PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
export NOVA3R_DIR=nova3r
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun "$PY" -u scripts/compare_three_latent_umap.py \
    --adapter-ckpt checkpoints/fullrec_nf16_vel03_complete_best.pt \
    --data-root /storage/group/cvpr/chwe/da3_nova3r/replica \
    --room office4 \
    --max-windows 25 \
    --out-dir outputs/latent_umap/office4_replica_complete
