#!/bin/bash
# =====================================================================
# SLURM job — Imagenette image control run on OFFICIAL DINOv2 (ViT-S).
#
# This runs on top of commit A (repo == official facebookresearch/dinov2). The
# only additions vs official are: an ImageFolder dataset reader + this config +
# this script. The learning pipeline (model, DINO/iBOT/KoLeo losses,
# DataAugmentationDINO multi-crop, collate, sampler, training loop) is official.
#
# Goal: confirm the loss clearly DECREASES on natural images (real augmentation
# gap between the two global crops). If it does, the machinery is correct and the
# fMRI flat loss is confirmed to be the masking-only (identical-views) design.
#
# Launch:  sbatch -A arieljaffe imagenette_test/train.sh
# Watch:   tail -f imagenette_test/logs/imagenette.out | grep total_loss
# =====================================================================

#SBATCH --job-name=dino-imagenette
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_DIR/runs/imagenette/$RUN_NAME}"
DATA_DIR="${DATA_DIR:-$LAB_DIR/data/imagenette}"
IMAGENETTE_URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
TRAIN_ROOT="$DATA_DIR/imagenette2-320/train"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOME="$LAB_DIR"                               # /sci/home not writable on nodes
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TASK_DIR/logs" "$DATA_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

LOG_OUT="$TASK_DIR/logs/${RUN_NAME}.out"
LOG_ERR="$TASK_DIR/logs/${RUN_NAME}.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

# --- download Imagenette once (idempotent) ------------------------------------
if [ ! -d "$TRAIN_ROOT" ]; then
    echo "Imagenette not found -> downloading to $DATA_DIR"
    ( cd "$DATA_DIR" && wget -q --show-progress -O imagenette2-320.tgz "$IMAGENETTE_URL" \
      && tar -xzf imagenette2-320.tgz && rm -f imagenette2-320.tgz )
fi
[ -d "$TRAIN_ROOT" ] || { echo "ERROR: $TRAIN_ROOT missing after download"; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
    # Tiny run (10 iters). warmup MUST shrink too, else warmup_iters > total_iters
    # makes CosineScheduler assert (len(schedule) != total_iters). So drop warmup
    # to 0 and set teacher-temp warmup to 1 epoch (=10 iters = total), no last-layer freeze.
    EXTRA="optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=10 optim.warmup_epochs=0 teacher.warmup_teacher_temp_epochs=1 optim.freeze_last_layer_epochs=0"
fi
# OVERFIT_N=k -> OVERFIT SANITY TEST: train on only k images (dataset prints them).
# A healthy loop MUST drive the loss down. Config is tuned to memorise fast:
#   batch = k (one batch = the k images, no within-batch repeats),
#   koleo=0 (repeated/similar CLS -> log(0) -> NaN otherwise),
#   short warmup + higher LR + no last-layer freeze so it learns from step 0.
if [ -n "${OVERFIT_N:-}" ]; then
    export IMAGENETTE_OVERFIT_N="$OVERFIT_N"
    EXTRA="$EXTRA train.batch_size_per_gpu=${OVERFIT_N} dino.koleo_loss_weight=0 \
optim.warmup_epochs=0.2 optim.base_lr=0.01 optim.freeze_last_layer_epochs=0 \
teacher.warmup_teacher_temp_epochs=2 optim.epochs=30 train.OFFICIAL_EPOCH_LENGTH=100"
fi
[ -n "${OVERRIDES:-}" ] && EXTRA="$EXTRA ${OVERRIDES}"

echo "============================================================"
echo "  DINOv2 Imagenette control   Node: $(hostname)   Date: $(date)"
echo "  venv:      $VENV"
echo "  config:    $CONFIG"
echo "  data:      $TRAIN_ROOT"
echo "  output:    $OUTPUT_DIR"
echo "  extra:     $EXTRA"
echo "============================================================"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$TRAIN_ROOT" \
    $EXTRA

echo ""
echo "Imagenette control run done: $(date)"
