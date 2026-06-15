#!/bin/bash
# =====================================================================
# SLURM job — download + downsample AOMIC PIOP1+PIOP2 rs-fMRI.
#
# Source: OpenNeuro (ds002785 = PIOP1, ds002790 = PIOP2).
# Open access — no DUA, no AWS credentials needed.
#
# Streams: download fMRIPrep-preprocessed MNI rest BOLD from s3://openneuro.org,
# downsample to (45, 54, 45) via trilinear, save .pt, delete raw.
# ~5-10 min per session (480 timepoints).
#
# Usage:
#   sbatch tasks/download_aomic/download_aomic.sh
#   LIMIT=5 sbatch tasks/download_aomic/download_aomic.sh           # test 5 subjects per dataset
#   DATASETS=piop1 sbatch tasks/download_aomic/download_aomic.sh    # PIOP1 only
# =====================================================================

#SBATCH --job-name=aomic-dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/download_aomic"
export AOMIC_DIR="$LAB_DIR/AOMIC_data"
export VENV_DIR="$LAB_DIR/torch_env"
export TMPDIR="$LAB_DIR/tmp/aomic_raw"
export PYTHONUNBUFFERED=1

mkdir -p "$TASK_DIR/logs"
mkdir -p "$AOMIC_DIR/downsampled"
mkdir -p "$TMPDIR"

LOG_OUT="$TASK_DIR/logs/download_aomic.out"
LOG_ERR="$TASK_DIR/logs/download_aomic.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/download_aomic_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/download_aomic_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  AOMIC PIOP1+PIOP2 download + downsample (trilinear -> 45x54x45)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  TASK_DIR:  $TASK_DIR"
echo "  AOMIC_DIR: $AOMIC_DIR/downsampled"
echo "  TMPDIR:    $TMPDIR"
echo "============================================================"

# Install missing pkgs (idempotent)
for pkg in boto3 nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

EXTRA_ARGS=""
[ -n "${LIMIT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
[ -n "${DATASETS:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --datasets $DATASETS"

cd "$OFFICIAL_DIR"
python "$TASK_DIR/download_aomic.py" \
    --output-dir "$AOMIC_DIR/downsampled" \
    --tmp-dir "$TMPDIR" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $AOMIC_DIR/downsampled"
echo "============================================================"
