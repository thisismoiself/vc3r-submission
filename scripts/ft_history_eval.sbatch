#!/bin/bash
#SBATCH --job-name=ft_hist
#SBATCH --partition=NORMAL
#SBATCH --qos=external
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --prefer="GPU_CC:9.0"
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/prakt/s0016/vc3r/experiments/overfit_8frames/ft_history_eval_%j.log
# Re-evaluate the Replica FINE-TUNE checkpoint history on the REAL whole-room office4 metric
# (the crop metric we selected by was misleading). One job, sequential stitches at 50 DDIM steps
# (consistent across all). Anchors: Stage-A warm-start (step 0) then FT steps 1k..11k.
PY=/usr/prakt/s0016/miniconda3/envs/nova3r/bin/python
cd /usr/prakt/s0016/vc3r
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
NPZ=outputs/tsdf_cache_replica/replica_office4.npz
OUT=/tmp/claude-264016/-usr-prakt-s0016-vc3r/c4ec2add-7077-4393-96b8-166d8c50fd94/scratchpad/ft_hist
H=outputs/consecutive_windows/snapshots/history_replica_ft
declare -A CKPTS=(
  [00k_stageA]=outputs/consecutive_windows/snapshots/stageA_v2_step78k_F80.pt
  [01k]=$H/stageA_step1k.pt [02k]=$H/stageA_step2k.pt [04k]=$H/stageA_step4k.pt
  [06k]=$H/stageA_step6k.pt [08k]=$H/stageA_step8k.pt [11k]=$H/stageA_step11k.pt
)
for tag in 00k_stageA 01k 02k 04k 06k 08k 11k; do
  ck=${CKPTS[$tag]}
  echo "########## FT-HIST $tag  ($ck) ##########"
  $PY scripts/stitch_room_diffusion.py --npz $NPZ --ckpt $ck --out-dir $OUT/$tag --win 128 --stride 64 --steps 50
done
echo "FT HISTORY EVAL DONE"
