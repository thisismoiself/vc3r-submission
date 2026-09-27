#!/bin/bash
#SBATCH --job-name=spp_eval
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/spp_eval_%j.log
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
$PY scripts/eval_scannetpp.py --ckpt "$CKPT" --cache-dir ${CACHE:-outputs/tsdf_cache_scannetpp_heldout} \
  --n-scenes ${NSC:-6} --which last --tag "$TAG"
echo "SPP EVAL DONE"
