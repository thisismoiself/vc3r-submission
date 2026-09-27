#!/usr/bin/env bash
#SBATCH --job-name=resflow_mw
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/resflow_mw_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/resflow_mw_%j.out
#
# Residual flow-matching head, MULTI-WINDOW. Gating one-window sanity PASSED: the flow
# closed 57% of the pred->oracle decode gap (2.09cm -> 0.89cm) on one window.
#
# This run tests the failure mode the investigation summary documented: the residual
# sampler collapsed to regressor level when amortised over ~2400 windows. Decode-eval runs
# on BOTH train and val windows every 2000 steps, so we see directly whether it (a) helps
# on train but collapses on val (overfit / information wall at scale), or (b) generalises.
# Success = val decode-chamfer-to-oracle drops below the pred baseline.
#
# Frozen base = fullrec_nf16_vel03_best.pt (the 0.371 FURN F@2 model). Head is tiny
# (~2.6M params); the cost is the 1225-window bank build (one frozen adapter forward each).
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/train_residual_flow.py \
  --base-ckpt "$OUT/fullrec_nf16_vel03_best.pt" \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
  --val-rooms $C/office4 \
  --max-per-root 175 --max-val-per-root 8 \
  --steps 12000 --batch-size 32 --lr 3e-4 \
  --hidden 512 --depth 4 --flow-steps 6 \
  --eval-every 2000 --eval-windows 6 --num-queries 20000 \
  --ckpt-out "$OUT/residual_flow_mw_head.pt" \
  > experiments/overfit_8frames/residual_flow_mw.log 2>&1
echo "=== [$(date)] RESIDUAL FLOW MW DONE exit=$? ==="
grep "decode-eval" experiments/overfit_8frames/residual_flow_mw.log | tail -8
