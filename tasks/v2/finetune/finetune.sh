#!/bin/bash
# =====================================================================
# SLURM job — fine-tune (SFT) a pretrained encoder on a downstream task
# (point 4, Brain-JEPA's headline protocol). Unfreezes the encoder and
# trains end-to-end + a linear head, leakage-free 70:30, mean±std over seeds.
#
# Usage (RUN can be a v1 run via RUNS_DIR, or a v2 run):
#   RUN=base   DATASET=ADNI DEPTH=last3 sbatch -A arieljaffe tasks/v2/finetune/finetune.sh
#   RUN=base   DATASET=ADNI DEPTH=all   SEEDS=0,1,2,3,4 sbatch -A arieljaffe tasks/v2/finetune/finetune.sh
#   RUNS_DIR=$LAB/runs/v1 RUN=base DATASET=HCP DEPTH=last3 sbatch -A arieljaffe tasks/v2/finetune/finetune.sh
#   SMOKE=1 RUN=base DATASET=ADNI sbatch -A arieljaffe tasks/v2/finetune/finetune.sh   # pipeline check
# =====================================================================

#SBATCH --job-name=ft
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v2/finetune"
VENV_DIR="$LAB_DIR/torch_env"

RUN="${RUN:-base}"
DATASET="${DATASET:-ADNI}"
DEPTH="${DEPTH:-last3}"                       # head | last3 | all
SEEDS="${SEEDS:-0}"
CKPT="${CKPT:-model_final.rank_0.pth}"
# base/fourier live in runs/v1; noblock2/pool in runs/v2 -> override RUNS_DIR as needed.
RUNS_DIR="${RUNS_DIR:-$LAB_DIR/runs/v2}"
RUN_DIR="$RUNS_DIR/$RUN"

mkdir -p "$TASK_DIR/logs"
SMOKE_TAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_TAG="_smoke"
LOG="$TASK_DIR/logs/ft_${RUN}_${DATASET}_${DEPTH}${SMOKE_TAG}.out"
exec >"$LOG" 2>&1

export TMPDIR="$LAB_DIR/tmp"
export HOME="$LAB_DIR"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Fine-tune  run=$RUN dataset=$DATASET depth=$DEPTH seeds=$SEEDS smoke=${SMOKE:-0}  Node=$(hostname)  $(date)"
srun python -u tasks/v2/finetune/finetune.py \
    --run-dir "$RUN_DIR" --dataset "$DATASET" --depth "$DEPTH" \
    --seeds "$SEEDS" --checkpoint "$CKPT" ${SMOKE:+--smoke}
echo "Done: $(date)"
