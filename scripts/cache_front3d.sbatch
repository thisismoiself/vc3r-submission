#!/bin/bash
#SBATCH --job-name=front3d_cache
#SBATCH --partition=NORMAL
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --qos=external
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/front3d_cache_%j.log
# CPU-only: assemble 3D-FRONT rooms -> cached TSDF grids. No GPU needed.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore
# pin per-process threads so N pool workers don't each spawn N threads (KD-tree/BLAS/Open3D)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
# 3000 houses ~ 7000 rooms; 18 parallel workers @ ~28s/room -> ~3 h.
$PY scripts/front3d_assembler.py \
  --out-dir outputs/tsdf_cache_front3d \
  --n-houses 3000 --start 0 --n-frames 24 --procs 1
echo "CACHE DONE"
