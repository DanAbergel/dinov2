#!/bin/bash
# =====================================================================
# SLURM job — V3 DEBUG / SANITY experiments on the fMRI foundation model.
#
# Same model + config as v2 (dinov2/configs/train/fmri_vits.yaml), but this
# script is for DIAGNOSTIC runs, NOT full pretraining:
#   - NO auto-probe (pure training + loss inspection).
#   - Meant to be paired with OVERFIT_N / UNFROZEN / OVERRIDES to isolate why
#     the loss stays flat (~10.3) on the full corpus.
#
# The overfit sanity check (memorize k scans; a healthy model drives the loss
# toward 0 — if not, learning is mechanically broken, not a data issue). The
# dataset prints exactly which scans it kept (dataset + subject + path):
#   RUN=overfit1 OVERFIT_N=1 UNFROZEN=1 \
#     OVERRIDES="optim.base_lr=8e-3 optim.warmup_epochs=0.1 optim.epochs=6 \
#                train.OFFICIAL_EPOCH_LENGTH=500 optim.freeze_last_layer_epochs=0 \
#                optim.min_lr=1e-3" \
#     sbatch -A arieljaffe tasks/v3/train/train.sh
#
# Watch it:
#   tail -f tasks/v3/train/logs/overfit1.out | grep total_loss
# =====================================================================

#SBATCH --job-name=fmri-v3-debug
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v3/train"
CONFIG="$OFFICIAL_DIR/dinov2/configs/train/fmri_vits.yaml"

# --- cluster-specific (override on the command line) ---------------------
VENV="${VENV:-$LAB_DIR/torch_env}"                 # dinov2 training env
RUN_NAME="${RUN_NAME:-${RUN:-debug}}"              # accept RUN or RUN_NAME
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_DIR/runs/v3/$RUN_NAME}"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# $HOME (/sci/home) is not writable on compute nodes; Triton/xformers JIT-compile
# CUDA kernels and cache them under ~/.triton -> PermissionError. Point HOME and
# the JIT caches at the writable lab dir.
export HOME="$LAB_DIR"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

LOG_OUT="$TASK_DIR/logs/${RUN_NAME}.out"
LOG_ERR="$TASK_DIR/logs/${RUN_NAME}.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

EXTRA=""
# SMOKE=1 -> tiny run to validate the pipeline (build, 1 epoch x 10 iters).
if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=10 evaluation.eval_period_iterations=0"
fi
# Optional batch override. With OVERFIT_N=1 only HCP remains -> sum(quota)=4, so
# batch_size_per_gpu must divide 4 (the config default 2 does). Not needed normally.
[ -n "${BATCH_PER_GPU:-}" ] && EXTRA="$EXTRA train.batch_size_per_gpu=${BATCH_PER_GPU}"
[ -n "${GRAD_ACCUM:-}" ]    && EXTRA="$EXTRA optim.grad_accum_steps=${GRAD_ACCUM}"
# UNFROZEN=1 -> freeze policy A (none): ALL transformer layers train during SSL.
[ "${UNFROZEN:-0}" = "1" ]  && EXTRA="$EXTRA optim.freeze_pretrained=none"
# OVERFIT_N=k -> DEBUG: train on only the first k scans (dataset prints their paths).
# Exported so the dataset (fmri_data.py) sees it through srun.
[ -n "${OVERFIT_N:-}" ]     && export FMRI_OVERFIT_N="${OVERFIT_N}"
# FIXED_WINDOW=1 -> DEBUG: pin the temporal window start to 0 so every load returns the
# EXACT same window (no temporal augmentation) — strictly identical input for overfit.
[ "${FIXED_WINDOW:-0}" = "1" ] && export FMRI_FIXED_WINDOW=1
# OVERRIDES -> any extra dinov2 config overrides, space-separated.
[ -n "${OVERRIDES:-}" ]     && EXTRA="$EXTRA ${OVERRIDES}"

echo "============================================================"
echo "  fMRI V3 DEBUG run   Node: $(hostname)   Date: $(date)"
echo "  venv:      $VENV"
echo "  config:    $CONFIG"
echo "  output:    $OUTPUT_DIR"
echo "  overfit_n: ${OVERFIT_N:-0}   smoke: ${SMOKE:-0}"
echo "  extra:     $EXTRA"
echo "============================================================"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Launch with srun (dinov2's distributed.enable() takes the SLURM path).
srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA

echo ""
echo "Training done: $(date)"
# NO probe here — this is a diagnostic run (loss inspection only).
