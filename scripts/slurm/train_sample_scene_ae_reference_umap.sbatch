#!/usr/bin/env bash
#SBATCH --job-name=train_ref_umap
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/usr/prakt/s0017/vc3r/logs/eval_niko/train_ref_umap_%j.out
#SBATCH --error=/usr/prakt/s0017/vc3r/logs/eval_niko/train_ref_umap_%j.out

set -euo pipefail
cd /usr/prakt/s0017/vc3r

PY=/usr/prakt/s0017/miniconda3/envs/vc3r-scripts/bin/python
OUT=outputs/latent_umap/office0_training_window
PLOTS="$OUT/scene_ae_reference"
export NOVA3R_DIR=nova3r
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun "$PY" -u scripts/compare_three_latent_umap.py \
    --adapter-ckpt checkpoints/fullrec_nf16_vel03_complete_best.pt \
    --data-root /storage/group/cvpr/chwe/da3_nova3r/replica \
    --room office0 \
    --max-windows 1 \
    --max-tokens-per-source 768 \
    --out-dir "$OUT"

srun "$PY" -u scripts/plot_scene_ae_reference_umap.py \
    --latents "$OUT/latents.npz" \
    --window 0 \
    --out-dir "$PLOTS"
