#!/bin/bash
# =====================================================================
# SLURM job — downsample already-downloaded OASIS-3 raw NIfTIs to .pt.
#
# Runs ONLY the downsample step (no download). Use this to re-process
# the raw_nifti/ folder after the download step has run, e.g. to test
# a fix to downsample_oasis3.py without re-downloading.
#
# Usage:
#   sbatch tasks/data_prep/download_oasis3/downsample_oasis3.sh
#   INPUT_DIR=/some/raw OUTPUT_DIR=/some/out sbatch tasks/data_prep/download_oasis3/downsample_oasis3.sh
# =====================================================================

#SBATCH --job-name=oasis3-ds
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/download_oasis3"
OASIS3_DIR="$LAB_DIR/OASIS3_data"
VENV_DIR="$LAB_DIR/torch_env"

INPUT_DIR="${INPUT_DIR:-$OASIS3_DIR/raw_nifti}"
OUTPUT_DIR="${OUTPUT_DIR:-$OASIS3_DIR/downsampled}"

mkdir -p "$TASK_DIR/logs" "$OUTPUT_DIR"

LOG_OUT="$TASK_DIR/logs/downsample_oasis3.out"
LOG_ERR="$TASK_DIR/logs/downsample_oasis3.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/downsample_oasis3_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/downsample_oasis3_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  OASIS-3 downsample-only"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  INPUT_DIR:  $INPUT_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "============================================================"

for pkg in nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

# DELETE_RAW=1 frees each raw dir right after its .pt is saved (the TR is
# captured to the manifest first). Use this when disk is tight.
EXTRA_ARGS=""
[ "${DELETE_RAW:-0}" = "1" ] && EXTRA_ARGS="--delete-raw"

python3 "$TASK_DIR/downsample_oasis3.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
