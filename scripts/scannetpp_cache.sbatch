#!/bin/bash
#SBATCH --job-name=spp_cache
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/spp_cache_%j.log
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
$PY scripts/scannetpp_assembler.py \
  --split /storage/group/dataset_mirrors/01_incoming/scannetpp/splits/nvs_sem_train.txt \
  --out-dir outputs/tsdf_cache_scannetpp --limit 50 --n-frames 50 --procs 8
echo "SPP CACHE DONE"
