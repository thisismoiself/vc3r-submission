#!/bin/bash
#SBATCH --job-name=front3d_v2
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=node1,node2,node6,node7
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/front3d_v2_%j.log
# Stage-A v2 RECACHE (CPU-only, co-runs under the 128G/20cpu QOS cap next to the retrain):
#   band 0.06m (3-vox truncation, paper)  +  dense room-sweep trajectory (grid x yaw coverage,
#   n-frames 60)  +  MORE rooms.  free-flood is DELIBERATELY OFF: on a non-watertight voxel shell
#   it leaked through 1-vox gaps and zeroed `unknown` (no targets) -> the frontier-restricted loss
#   + RePaint replace handle far-field hallucination instead. Stores per-npz `band` so training can
#   normalize 0.06 correctly. iter_rooms skips cached npzs PRE-assembly, so resumes are cheap.
#   --exclude steers off the oldest titan/p6000 nodes onto a modern-CPU node.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
$PY scripts/front3d_assembler.py \
  --out-dir outputs/tsdf_cache_front3d_v2 \
  --n-houses 6000 --start 0 --n-frames 60 \
  --band 0.06 --procs 2
echo "CACHE v2 DONE"
