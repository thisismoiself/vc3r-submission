#!/bin/bash
#SBATCH --job-name=v5_sweep
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/v5_sweep_%j.log
# Batch-eval every preserved v5 ladder checkpoint -> full early F-vs-step curve in one cluster job
# (fast GPU), instead of chasing gated per-point evals. Writes step,F to v5_F_history_sweep.csv.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
OUT=outputs/v5_F_history_sweep.csv
echo "step,cross_room_F" > "$OUT"
for ck in $(ls outputs/consecutive_windows/snapshots/stageA_v5_ladder_*.pt | sort -t_ -k4 -n); do
  step=$(echo "$ck" | sed 's/.*ladder_//;s/.pt//')
  od=experiments/overfit_8frames/v5_sweep_s${step}
  rm -rf "$od"
  $PY scripts/progress_strip.py --band 0.06 --ckpt "$ck" \
    --cache-dir outputs/tsdf_cache_front3d_v2_holdout --val-frac 1.0 \
    --room-idx 0 --track-rooms 0 1 2 3 4 --steps 100 \
    --best-ckpt experiments/overfit_8frames/_sweep_junk.pt --best-json experiments/overfit_8frames/_sweep_junk.json --out-dir "$od" > "${od}.log" 2>&1
  F=$(grep -oE "cross-room F=[0-9.]+" "${od}.log" | tail -1 | grep -oE "[0-9.]+")
  echo "${step},${F:-ERR}" >> "$OUT"
  echo "swept step $step -> F=${F:-ERR}"
  rm -rf "$od"
done
echo "V5 SWEEP DONE"
