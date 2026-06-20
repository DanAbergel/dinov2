#!/bin/bash
# =====================================================================
# SLURM job — convert Sagi's stacked ADNI tensor into per-scan .pt files
# that match the rest of the corpus (HCP / ABIDE / OASIS-3 / AOMIC).
#
# Reads:
#   $SOURCE/all_4d_downsampled.pt   (N, X, Y, Z, T)
#   $SOURCE/index_to_name.json
#   $SOURCE/imageID_to_labels.json
# Writes:
#   $OUTPUT_DIR/<subject_id>/<image_id>.pt   shape (T, 45, 54, 45)
#   $OUTPUT_DIR/adni_manifest.csv            (labels incl. degradation_binary_1/2/3)
#
# One-shot, CPU-only (no GPU needed). Usage:
#   sbatch tasks/convert_adni/convert_adni.sh
# =====================================================================

#SBATCH --job-name=convert-adni
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/convert_adni"
VENV_DIR="$LAB_DIR/torch_env"

SOURCE="${SOURCE:-/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_DIR/ADNI_data/downsampled}"
TR="${TR:-3.0}"

mkdir -p "$TASK_DIR/logs" "$OUTPUT_DIR"

LOG_OUT="$TASK_DIR/logs/convert_adni.out"
LOG_ERR="$TASK_DIR/logs/convert_adni.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/convert_adni_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/convert_adni_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  Convert ADNI (Sagi stacked tensor -> per-scan .pt)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  SOURCE:     $SOURCE"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  TR assumed: $TR"
echo "============================================================"

python3 "$TASK_DIR/convert_adni.py" \
    --source "$SOURCE" \
    --output-dir "$OUTPUT_DIR" \
    --tr "$TR"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
