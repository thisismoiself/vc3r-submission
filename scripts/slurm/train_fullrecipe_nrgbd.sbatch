#!/usr/bin/env bash
#SBATCH --job-name=fr_nrgbd
#SBATCH --partition=NORMAL
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fr_nrgbd_%j.out
#SBATCH --error=/usr/prakt/s0016/vc3r/scripts/slurm/logs/fr_nrgbd_%j.out
#
# DATA-SCALING A/B: the proven full recipe (plain Hungarian MSE + velocity 0.3, nf16
# span24-100) trained on Replica-7 + NeuralRGBD-9 instead of Replica-7 alone.
#   baseline: 1225 windows /  7 scenes -> FURN F@2 0.371
#   this run: ~2185 windows / 16 scenes  (+78% windows, 2.3x scene diversity)
#
# EXACTLY ONE VARIABLE CHANGES: the data. Architecture, loss, schedule, steps, batch,
# LR, dropout and val set are byte-identical to scripts/slurm/train_fullrecipe.sbatch.
#   * Architecture unchanged on purpose: the token-noise sensitivity result says the
#     adapter's error does ~6x the geometric damage of isotropic noise at the same MSE,
#     i.e. the error is STRUCTURED, not a capacity ceiling. More params don't fix signal.
#   * steps stays 18000 (= exactly 2 cosine cycles: 6000 + 12000). This is a
#     same-compute/more-data A/B; extend only if the run is still improving at 18k.
#   * NO --geom-cond here: it needs GT geometry at deploy (pts_norm 0.403) and the
#     GT-free DA3 variant collapses to ~0.22. Separate axis, revisit after.
#   * val stays office4 so every number is directly comparable to 0.371.
#
# --mem=120G: 1225 windows needed ~41G RAM; 2185 scales to ~73G (measured 17.2MB/window
# resident x ~1.95 overhead). Per-user QOS caps mem at 128G, so 120G is the practical
# ceiling -- still ~1.6x the requirement, safe against the OOM-kill that bit us locally.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
export NOVA3R_DIR=nova3r_lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=scripts/data/fc_nf16_span24_100_l13
N=scripts/data/nrgbd_nf16_span24_100_l13
OUT=outputs/consecutive_windows; mkdir -p "$OUT"
echo "=== [$(date)] host=$(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "windows: replica=$(ls -d $C/office0/*/ $C/office1/*/ $C/office2/*/ $C/office3/*/ $C/room0/*/ $C/room1/*/ $C/room2/*/ 2>/dev/null | wc -l) nrgbd=$(ls -d $N/*/*/ 2>/dev/null | wc -l)"

$PY scripts/train_online_var_hungarian.py \
  --rooms $C/office0 $C/office1 $C/office2 $C/office3 $C/room0 $C/room1 $C/room2 \
          $N/breakfast_room $N/complete_kitchen $N/green_room $N/grey_white_room \
          $N/kitchen $N/morning_apartment $N/staircase $N/thin_geometry $N/whiteroom \
  --val-rooms $C/office4 --max-per-root 175 --max-val-per-root 60 \
  --steps 18000 --batch-size 24 --lr 5e-4 --lr-min 1e-6 --warmup-steps 1000 \
  --cosine-t0 6000 --cosine-t-mult 2 --ema-decay 0.999 \
  --drop 0.2 --token-drop 0.2 --weight-decay 1e-4 --cache-dtype float16 --log-every 200 \
  --loss-mode plain --vel-loss-weight 0.3 --vel-points 1024 \
  --ckpt-out "$OUT/fullrec_nrgbd_best.pt" \
  > experiments/overfit_8frames/fullrec_nrgbd.log 2>&1
echo "=== [$(date)] TRAIN DONE exit=$? ==="

# --- auto stitch eval on held-out office4, same allocation (no GPU contention) ---
CK="$OUT/fullrec_nrgbd_best.pt"
if [ -f "$CK" ]; then
  echo "=== [$(date)] running stitch eval on $CK ==="
  $PY scripts/stitch_office4.py --ckpt "$CK" --fm-sampling midpoint --num-queries 50000 \
      --out-tag fullrec_nrgbd_midpoint \
      > experiments/overfit_8frames/eval_fullrec_nrgbd_stitch.log 2>&1
  echo "=== [$(date)] EVAL DONE exit=$? ==="
  $PY -c "import json; m=json.load(open('outputs/replica/stitch_office4_fullrec_nrgbd_midpoint/metrics.json')); p=m['pred']; pf=m['pred_furniture']; print(f\"NRGBD-DATA: whole {p['chamfer_m']:.4f} wF@1 {p['F@1cm']:.3f} wF@5 {p['F@5cm']:.3f} | FURN F@1 {pf['F@1cm']:.4f} F@2 {pf['F@2cm']:.4f} F@5 {pf['F@5cm']:.4f}\"); print('baseline (replica-only): whole 0.0567 | FURN F@2 0.3709'); print('oracle ceiling:          whole 0.0440 | FURN F@2 0.5510')"
else
  echo "=== [$(date)] NO CKPT — training failed ==="
fi
