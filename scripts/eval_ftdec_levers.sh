#!/bin/bash
# Geometric (FURN F@2) head-to-head on office4 to separate each effect:
#   base ref (known 0.333, bestcombo adapter + BASE decoder)
#   -> bestcombo adapter + LoRA decoder      = decoder-swap effect
#   -> new ftdec adapter  + LoRA decoder      = retrain effect (matched train/eval decoder)
#   -> + importance + filter                  = decode-time levers
cd /usr/prakt/s0016/vc3r
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
LORA=outputs/consecutive_windows/diag_decoder_lora_head.pt
FTDEC=outputs/consecutive_windows/complete_ftdec_a40_pca32_best.pt
BEST=outputs/consecutive_windows/complete_bestcombo_pca32_best.pt

run () {  # tag ckpt extra-args...
  tag=$1; ckpt=$2; shift 2
  echo "==================== RUN $tag ===================="
  date
  $PY scripts/stitch_office4.py --ckpt "$ckpt" --complete-target \
      --decoder-ckpt "$LORA" --fm-sampling midpoint --num-queries 50000 "$@" --out-tag "$tag" 2>&1
  echo "==================== DONE $tag ===================="
}

run ftdec_lora_q50        "$FTDEC"
run bestcombo_lora_q50     "$BEST"
run ftdec_lora_impfilt_q50 "$FTDEC" --importance --filter-sor
echo "ALLDONE"
