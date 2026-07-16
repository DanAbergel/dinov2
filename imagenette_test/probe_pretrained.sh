#!/bin/bash
# =====================================================================
# SLURM job — linear probe on the OFFICIAL pretrained DINOv2 ViT-S/14.
# Downloads the published LVD-142M weights, extracts frozen CLS features on
# Imagenette train+val, fits a logistic regression, prints val metrics.
# This is the "real DINO" reference (top of the spectrum) vs our tiny run.
#
# Launch:  sbatch -A arieljaffe imagenette_test/probe_pretrained.sh
# Watch:   tail -f imagenette_test/logs/probe_pretrained.out
# =====================================================================

#SBATCH --job-name=dino-pretrained-probe
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
REPO_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$REPO_DIR/imagenette_test"

VENV="${VENV:-$LAB_DIR/torch_env}"
DATA_DIR="${DATA_DIR:-$LAB_DIR/data/imagenette}"
TRAIN_ROOT="$DATA_DIR/imagenette2-320/train"
VAL_ROOT="$DATA_DIR/imagenette2-320/val"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TORCH_HOME="$LAB_DIR/cache/torch"          # where pretrained weights are cached
export HOME="$LAB_DIR"
mkdir -p "$TMPDIR" "$TASK_DIR/logs" "$TORCH_HOME"

LOG_OUT="$TASK_DIR/logs/probe_pretrained.out"
LOG_ERR="$TASK_DIR/logs/probe_pretrained.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

[ -d "$VAL_ROOT" ] || { echo "ERROR: val data $VAL_ROOT not found"; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "  DINOv2 PRETRAINED probe   Node: $(hostname)   Date: $(date)"
echo "  weights cache: $TORCH_HOME"
echo "  train/val: $TRAIN_ROOT | $VAL_ROOT"
echo "============================================================"

python "$TASK_DIR/probe_pretrained.py" \
    --train-root "$TRAIN_ROOT" \
    --val-root "$VAL_ROOT"

echo ""
echo "Pretrained probe done: $(date)"
