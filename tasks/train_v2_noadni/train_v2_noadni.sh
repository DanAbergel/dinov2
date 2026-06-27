#!/bin/bash
# =====================================================================
# SLURM job — V2 pretraining with ADNI FULLY EXCLUDED (freezeC = C).
#
# Why: ADNI from Sagi is small (215 subjects). A leakage-free fixed-split
# probe leaves only ~32 test subjects -> too noisy to compare to SOTA
# (Brain-JEPA n=189). Excluding ALL ADNI from pretraining lets the
# downstream probe run subject-aware k-fold over the WHOLE ADNI cohort
# (encoder never saw any ADNI) -> stable, SOTA-scale ADNI comparison.
#
# Differences vs the main run:
#   train.dataset_path = Mixed:exclude=ADNI   (drop all ADNI from SSL)
#   optim.freeze_pretrained = fmri_only       (strategy C, our best config)
#
# After this finishes, probe with:
#   RUN=fmri_v2_freezeC_noADNI DATASET=ADNI KFOLD=5 \
#       sbatch -A arieljaffe tasks/probe/probe.sh
#
# Usage:
#   sbatch -A arieljaffe tasks/train_v2_noadni/train_v2_noadni.sh
#   SMOKE=1 sbatch -A arieljaffe tasks/train_v2_noadni/train_v2_noadni.sh
# =====================================================================

#SBATCH --job-name=fmri-v2-noadni
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/train_v2_noadni"
CONFIG="$OFFICIAL_DIR/dinov2/configs/train/fmri_vits.yaml"
VENV="${VENV:-$LAB_DIR/torch_env}"

RUN_NAME="${RUN_NAME:-fmri_v2_freezeC_noADNI}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_DIR/runs/$RUN_NAME}"

export TMPDIR="$LAB_DIR/tmp"
export HOME="$LAB_DIR"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR"

LOG_OUT="$TASK_DIR/logs/${RUN_NAME}.out"
LOG_ERR="$TASK_DIR/logs/${RUN_NAME}.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Bake in: exclude ALL ADNI + freeze policy C.
EXTRA="train.dataset_path=Mixed:exclude=ADNI optim.freeze_pretrained=fmri_only"
if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="$EXTRA optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=10 evaluation.eval_period_iterations=0"
fi

echo "============================================================"
echo "  fMRI V2 (no ADNI, freezeC)   Node: $(hostname)   Date: $(date)"
echo "  output: $OUTPUT_DIR"
echo "  extra:  $EXTRA"
echo "============================================================"

srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA

echo ""
echo "Done: $(date)"
