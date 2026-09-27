#!/usr/bin/env bash
# Export oracle pointclouds for office0 (train) and office4 (held-out), each with the BASELINE
# decoder and the generalized LoRA decoder, so the crispness change is visible side by side.
set -uo pipefail
cd /usr/prakt/s0016/vc3r
export HF_HOME=/usr/prakt/s0016/vc3r/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
ADP=outputs/consecutive_windows/fullrec_nf16_vel03_complete_best.pt
LORA=outputs/consecutive_windows/diag_decoder_lora_head.pt
LOG=experiments/overfit_8frames

run () {  # room  tag  [decoder-ckpt]
  local room=$1 tag=$2 dec=${3:-}
  local extra=""; [ -n "$dec" ] && extra="--decoder-ckpt $dec"
  echo "=== [$(date +%H:%M:%S)] stitch $room  tag=$tag  dec=${dec:-BASELINE} ==="
  $PY scripts/stitch_office4.py --ckpt $ADP --room "$room" --complete-target \
      --fm-sampling midpoint --num-queries 50000 --out-tag "$tag" $extra \
      > "$LOG/pc_${tag}.log" 2>&1
  echo "    oracle furniture:"; grep -A3 "ORACLE furniture-region" "$LOG/pc_${tag}.log" | grep -i "chamfer_m\|F@2cm"
  echo "    ply: $(ls outputs/replica/stitch_${room}_${tag}/${room}_oracle_stitched.ply 2>/dev/null)"
}

run office4 office4_basedec ""
run office4 office4_loradec "$LORA"
run office0 office0_basedec ""
run office0 office0_loradec "$LORA"
echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
