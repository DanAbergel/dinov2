#!/bin/bash
# =====================================================================
# SLURM job — linear probe on the Imagenette-trained DINOv2 backbone.
# Loads the run's checkpoint, extracts frozen CLS features on Imagenette
# train+val, fits a logistic regression, prints val top-1 accuracy.
#
# This is the REAL test of whether the model learned (the DINO loss curve
# cannot tell us). Point RUN_DIR at the training run's output dir.
#
# Launch:  sbatch -A arieljaffe imagenette_test/probe.sh
# Watch:   tail -f imagenette_test/logs/probe.out
# =====================================================================

#SBATCH --job-name=dino-imagenette-probe
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
CONFIG="$REPO_DIR/dinov2/configs/train/imagenette_vits.yaml"

VENV="${VENV:-$LAB_DIR/torch_env}"
RUN_NAME="${RUN_NAME:-${RUN:-imagenette}}"
RUN_DIR="${RUN_DIR:-$LAB_DIR/runs/imagenette/$RUN_NAME}"     # the TRAINING output dir (has the checkpoint)
DATA_DIR="${DATA_DIR:-$LAB_DIR/data/imagenette}"
TRAIN_ROOT="$DATA_DIR/imagenette2-320/train"
VAL_ROOT="$DATA_DIR/imagenette2-320/val"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export HOME="$LAB_DIR"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

LOG_OUT="$TASK_DIR/logs/probe.out"
LOG_ERR="$TASK_DIR/logs/probe.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

[ -d "$RUN_DIR" ] || { echo "ERROR: run dir $RUN_DIR not found"; exit 1; }
[ -d "$VAL_ROOT" ] || { echo "ERROR: val data $VAL_ROOT not found"; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "  DINOv2 Imagenette PROBE   Node: $(hostname)   Date: $(date)"
echo "  run dir (checkpoint): $RUN_DIR"
echo "  last_checkpoint: $(cat "$RUN_DIR/last_checkpoint" 2>/dev/null || echo '??')"
echo "  train/val: $TRAIN_ROOT | $VAL_ROOT"
echo "============================================================"

# PROBE_OPTS lets you pass config overrides so the model matches the checkpoint
# (e.g. dino.head_n_prototypes=128 ibot.head_n_prototypes=128 for a small-head run).
srun python "$TASK_DIR/probe.py" \
    --config-file "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --train-root "$TRAIN_ROOT" \
    --val-root "$VAL_ROOT" \
    ${PROBE_OPTS:-}

echo ""
echo "Probe done: $(date)"
