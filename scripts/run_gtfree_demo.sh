#!/bin/bash
# GT-free adapter demo: reconstruct one room and write point clouds you can open in a viewer.
#
# Usage:  bash scripts/run_gtfree_demo.sh <room> [data_root] [ckpt]
#   <room>       office4 | office0 | room1 ... (Replica)   |   breakfast_room (NeuralRGBD)   |   heads (7-Scenes)
#   [data_root]  default /usr/prakt/s0017/Replica
#                use /usr/prakt/s0017/NeuralRGBD          for breakfast_room
#                use /usr/prakt/s0017/SevenScenes_converted for heads
#   [ckpt]       default = Replica-only GT-free adapter
#                for NeuralRGBD/7-Scenes use the data-scaled one:
#                  outputs/consecutive_windows/da3pose_nrgbd_complete_holdout_bfr_best.pt
#
# The reconstruction is GT-FREE at the input (DA3 predicts the camera poses that condition
# the adapter, --da3pose-tokens). Placement/scale still use the GT trajectory, so the numbers
# are an upper bound. Prints Chamfer/F-scores and writes 3 .ply files.
set -euo pipefail
ROOM="${1:?usage: run_gtfree_demo.sh <room> [data_root] [ckpt]}"
DATA="${2:-/storage/group/cvpr/chwe/da3_nova3r/replica}"
CKPT="${3:-outputs/consecutive_windows/da3pose_nf16_vel03_complete_best.pt}"

cd /usr/prakt/s0017/vc3r
source /usr/prakt/s0017/miniconda3/etc/profile.d/conda.sh
conda activate vc3r-scripts
export PYTHONWARNINGS=ignore DA3_LOG_LEVEL=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/stitch_office4.py \
  --room "$ROOM" --data-root "$DATA" --ckpt "$CKPT" \
  --complete-target --da3pose-tokens \
  --n-frames 8 --stride 10 --fm-sampling midpoint --num-queries 50000 \
  --out-tag demo

OUT="outputs/replica/stitch_${ROOM}_demo"
echo
echo "Done. Outputs in ${OUT}/ :"
echo "  ${ROOM}_pred_stitched.ply    <- GT-free adapter reconstruction (this is 'ours')"
echo "  ${ROOM}_oracle_stitched.ply  <- NOVA3R oracle ceiling (decode of GT-derived z*)"
echo "  ${ROOM}_input_stitched.ply   <- input / visible geometry"
echo "Open the .ply files in MeshLab / CloudCompare / Open3D to inspect."
