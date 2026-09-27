#!/bin/bash
cd /usr/prakt/s0016/vc3r
# wait for the augmentation process to finish
while kill -0 1681711 2>/dev/null; do sleep 30; done
sleep 5
NSEG=$(ls outputs/tsdf_cache_replica_ft_train/replica_*_seg*.npz 2>/dev/null | wc -l)
NTOT=$(ls outputs/tsdf_cache_replica_ft_train/*.npz 2>/dev/null | wc -l)
echo "$(date) augmentation done: $NSEG seg variants, $NTOT total train caches"
if squeue -u s0016 -h -o "%j" 2>/dev/null | grep -q ft_replica; then
  echo "$(date) FT already present — not resubmitting"
else
  sbatch scripts/finetune_replica.sbatch
  echo "$(date) FT submitted (corrected loss + augmented data)"
fi
