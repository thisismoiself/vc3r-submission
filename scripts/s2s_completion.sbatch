#!/bin/bash
#SBATCH --job-name=s2s_comp
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/s2s_comp_%j.log
# Seen2Scene whole-room completion (drop_bbox) on our DA3-Replica scenes, on an A40 (48GB) — the local
# login-node GPU is saturated by other users. Uses the run_completion.sh core (TBB launcher + env vars).
# Submit ONLY when a SLURM slot is free (2-job cap): do NOT displace v5 training.
cd /usr/prakt/s0016/seen2scene_eval
for r in office4 office2 room1; do
  echo "########## Seen2Scene completion: $r ##########"
  bash run_completion.sh "$r"
done
echo "S2S COMPLETION DONE"
