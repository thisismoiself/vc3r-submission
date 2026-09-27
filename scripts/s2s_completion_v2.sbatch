#!/bin/bash
#SBATCH --job-name=s2s_v2
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/s2s_v2_%j.log
# Leaner Seen2Scene completion: office2 + room1 only (office4 already done), any big GPU
# (Ampere/Ada/Hopper, all >=48GB here), taller-Z canvas + mesh-only export (see run_completion_v2.sh).
# Does NOT displace v5 (separate slot).
cd /usr/prakt/s0016/seen2scene_eval
for r in office2 room1; do
  echo "########## Seen2Scene completion v2: $r ##########"
  bash run_completion_v2.sh "$r"
done
echo "S2S COMPLETION DONE"
