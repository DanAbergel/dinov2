#!/bin/bash
# =====================================================================
# SLURM job — download + downsample ABIDE I preprocessed rs-fMRI.
#
# Source: Preprocessed Connectomes Project on the public fcp-indi S3
# bucket (anonymous, no credentials). Pipeline=cpac, strategy=
# nofilt_noglobal (minimal — no bandpass, no GSR), derivative=func_preproc
# (4D BOLD already in MNI). Only spatial downsampling needed.
#
# Usage:
#   sbatch tasks/data_prep/download_abide/download_abide.sh
#   LIMIT=5 sbatch tasks/data_prep/download_abide/download_abide.sh           # test
#   DATASETS=abide1,abide2 sbatch tasks/data_prep/download_abide/download_abide.sh
# =====================================================================

#SBATCH --job-name=abide-dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/download_abide"
export ABIDE_DIR="$LAB_DIR/ABIDE_data"
export VENV_DIR="$LAB_DIR/torch_env"
export TMPDIR="$LAB_DIR/tmp/abide_raw"
export PYTHONUNBUFFERED=1

mkdir -p "$TASK_DIR/logs" "$ABIDE_DIR/downsampled" "$TMPDIR"

LOG_OUT="$TASK_DIR/logs/download_abide.out"
LOG_ERR="$TASK_DIR/logs/download_abide.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/download_abide_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/download_abide_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  ABIDE download + downsample (trilinear -> 45x54x45)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  ABIDE_DIR: $ABIDE_DIR/downsampled"
echo "============================================================"

for pkg in boto3 nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

EXTRA_ARGS=""
[ -n "${LIMIT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
[ -n "${DATASETS:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --datasets $DATASETS"

cd "$OFFICIAL_DIR"
python "$TASK_DIR/download_abide.py" \
    --output-dir "$ABIDE_DIR/downsampled" \
    --tmp-dir "$TMPDIR" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $ABIDE_DIR/downsampled"
echo "============================================================"
