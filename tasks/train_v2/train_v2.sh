#!/bin/bash
# =====================================================================
# SLURM job — train the V2 fMRI foundation model (official dinov2 fork).
#
# 5-source corpus (HCP/ABIDE/OASIS/AOMIC/ADNI, 4627 scans) with:
#   - online TR harmonization to 0.72s + T_fixed=270 window (MixedFMRIDataset)
#   - ProportionalBatchSampler (HCP4/ABIDE4/OASIS4/ADNI3/AOMIC1 = 16)
#   - masking-only augmentation (full-image crops + per-token random masking)
#   - learned spatial pos (Fourier is a LATER ablation)
#
# Config: dinov2/configs/train/fmri_vits.yaml
#
# IMPORTANT — verify these for your cluster before launching:
#   VENV  : the dinov2 env (xformers etc.), NOT the data-tasks torch_env.
#           Defaults to the in-repo dinov2/env_dino; override with VENV=...
#   GRES  : GPU type/count. Defaults to gpu:l40s:1; override with GRES=...
#
# Smoke test FIRST (cheap, ~10 iters, catches build/shape bugs):
#   SMOKE=1 sbatch tasks/train_v2/train_v2.sh
# Full run:
#   sbatch tasks/train_v2/train_v2.sh
# =====================================================================

#SBATCH --job-name=fmri-v2
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/train_v2"
CONFIG="$OFFICIAL_DIR/dinov2/configs/train/fmri_vits.yaml"

# --- cluster-specific (override on the command line) ---------------------
VENV="${VENV:-$LAB_DIR/torch_env}"                 # dinov2 training env
RUN_NAME="${RUN_NAME:-fmri_v2_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_DIR/runs/$RUN_NAME}"
MASTER_PORT="${MASTER_PORT:-29531}"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TASK_DIR/logs"

LOG_OUT="$TASK_DIR/logs/${RUN_NAME}.out"
LOG_ERR="$TASK_DIR/logs/${RUN_NAME}.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

# SMOKE=1 -> tiny run to validate the pipeline (build, 1 epoch x 10 iters).
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=10 evaluation.eval_period_iterations=0"
fi

echo "============================================================"
echo "  fMRI V2 training   Node: $(hostname)   Date: $(date)"
echo "  venv:    $VENV"
echo "  config:  $CONFIG"
echo "  output:  $OUTPUT_DIR"
echo "  smoke:   ${SMOKE:-0}   extra: $EXTRA"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
    dinov2/train/train.py \
        --config-file "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        $EXTRA

echo ""
echo "Done: $(date)"
