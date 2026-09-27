#!/bin/bash
# Submit the remaining complete-target loss-ablation arms as soon as the 2-job
# QOS submit cap frees up. Only submits arms whose best ckpt does not yet exist
# and that are not already queued/running. Distinct -J name per arm for detection.
set -u
cd /usr/prakt/s0016/vc3r
declare -a ARMS=("plain 0.1 plain_vel01" "plain 1.0 plain_vel10")
for a in "${ARMS[@]}"; do
  set -- $a; LOSS=$1; VEL=$2; TAG=$3
  CK=outputs/consecutive_windows/abl_complete_${TAG}_best.pt
  if [ -f "$CK" ]; then echo "SKIP $TAG (ckpt exists)"; continue; fi
  if squeue -u s0016 -h -o "%j" | grep -qx "abl_$TAG"; then echo "SKIP $TAG (already queued/running)"; continue; fi
  N=$(squeue -u s0016 -h -o "%j" | grep -c '^abl')
  if [ "$N" -ge 2 ]; then echo "WAIT $TAG (submit cap: $N jobs)"; continue; fi
  OUT=$(sbatch -J "abl_$TAG" scripts/abl_complete_loss.sbatch $LOSS $VEL $TAG 2>&1)
  echo "$OUT" | grep -iE "Submitted batch job|QOSMax|violates"
done
