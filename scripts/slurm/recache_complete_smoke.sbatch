#!/usr/bin/env bash
#SBATCH --job-name=recache_smoke
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/recache_smoke_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/recache_smoke_%j.out
set -uo pipefail
cd /usr/prakt/s0016/vc3r
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [$(date)] host=$(hostname) ==="
/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python -u scripts/recache_complete_target.py --rooms office4 --limit 2
echo "=== [$(date)] SMOKE DONE exit=$? ==="
ls -la scripts/data/fc_nf16_span24_100_l13_complete/office4/*/ 2>/dev/null | head -20
