#!/bin/bash
#SBATCH --job-name=repl_da3_006
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/repl_da3_006_%j.log
# 0.06m DA3 Replica caches so the v3 / DA3-sim (0.06m) models can be evaluated on real DA3 input.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 HF_HOME=/usr/prakt/s0016/vc3r/.hf_cache
$PY scripts/replica_da3_cache.py --rooms office0 office1 office2 office3 office4 room0 room1 room2 \
  --band 0.06 --out-dir outputs/tsdf_cache_replica_da3_006
echo "DA3 006 CACHE DONE"
