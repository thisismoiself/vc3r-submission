#!/bin/bash
#SBATCH --job-name=ftdec_a40
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/ftdec_a40_%j.log
#SBATCH --error=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/ftdec_a40_%j.log

# Full 18k-step retrain of the DA3->NOVA3R adapter with the LoRA-finetuned decoder
# folded into the velocity loss. Runs on an A40 (48GB) node with 1.5TB RAM, so tokens
# are held IN RAM (no --lazy-da3 streaming) -- both faster per step and an exact
# replica of the bestcombo recipe. DISTINCT ckpt name so it never collides with the
# local RTX-4060Ti run's checkpoints on shared NFS.
set -euo pipefail

REPO=/usr/prakt/s0016/vc3r
cd "$REPO"

source /usr/prakt/s0016/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

echo "[sbatch] host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[sbatch] cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE"

CACHE=scripts/data/fc_nf16_span24_100_l13_complete

srun python scripts/train_online_var_hungarian.py \
  --rooms $CACHE/office0 $CACHE/office1 $CACHE/office2 $CACHE/office3 $CACHE/room0 $CACHE/room1 $CACHE/room2 \
  --val-root $CACHE/office4 \
  --pca-project outputs/consecutive_windows/pca_basis_replica7_complete.pt --pca-dims 32 \
  --loss-mode plain \
  --vel-loss-weight 0.3 --vel-points 1024 \
  --decoder-head outputs/consecutive_windows/diag_decoder_lora_head.pt \
  --drop 0.2 --token-drop 0.2 --batch-size 24 \
  --steps 18000 --warmup-steps 1000 --lr 5e-4 --lr-min 1e-6 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 --weight-decay 1e-4 \
  --ckpt-every 2000 \
  --ckpt-out outputs/consecutive_windows/complete_ftdec_a40_pca32_best.pt

echo "[sbatch] DONE job=$SLURM_JOB_ID"
