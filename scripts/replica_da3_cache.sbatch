#!/bin/bash
#SBATCH --job-name=repl_da3
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/repl_da3_%j.log
# Build Replica caches whose PARTIAL is the real DA3 reconstruction (holes+noise); GT = clean mesh.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 HF_HOME=/usr/prakt/s0016/vc3r/.hf_cache
$PY scripts/replica_da3_cache.py --rooms ${ROOMS:-office4} --out-dir outputs/tsdf_cache_replica_da3
echo "REPLICA DA3 CACHE DONE"
