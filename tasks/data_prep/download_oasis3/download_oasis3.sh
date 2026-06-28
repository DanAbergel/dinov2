#!/bin/bash
# =====================================================================
# SLURM job — download + downsample OASIS-3 baseline rs-fMRI.
#
# Strategy: for each OASIS-3 subject, take the EARLIEST MR session
# containing a resting-state BOLD scan ("baseline visit") and download
# only that scan. Result: ~1000 unique-subject scans, ~85 GB downsampled.
#
# Prereqs on Moriah:
#   - NITRC-IR account (XNAT) with access to project OASIS3
#   - python pkgs: requests, nibabel (auto-installed via pip if missing)
#   - Password in ONE of:
#       a) env var XNAT_PASSWORD set before running sbatch
#       b) ~/.xnat_password file (chmod 600), read by the python script
#
# Username is passed via env var XNAT_USERNAME (defaults to 'danab').
#
# Usage:
#   # Method (a) — env var password:
#   export XNAT_PASSWORD='your-password'
#   sbatch tasks/data_prep/download_oasis3/download_oasis3.sh
#
#   # Method (b) — password file:
#   echo 'your-password' > ~/.xnat_password
#   chmod 600 ~/.xnat_password
#   sbatch tasks/data_prep/download_oasis3/download_oasis3.sh
#
#   # Test on 5 subjects:
#   LIMIT=5 sbatch tasks/data_prep/download_oasis3/download_oasis3.sh
# =====================================================================

#SBATCH --job-name=oasis3-dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/download_oasis3"
export OASIS3_DIR="$LAB_DIR/OASIS3_data"
export VENV_DIR="$LAB_DIR/torch_env"
export TMPDIR="$LAB_DIR/tmp/oasis3_raw"
export PYTHONUNBUFFERED=1

# Default username — override with XNAT_USERNAME=other sbatch ...
XNAT_USERNAME="${XNAT_USERNAME:-danab95}"

mkdir -p "$TASK_DIR/logs"
mkdir -p "$OASIS3_DIR/downsampled"
mkdir -p "$TMPDIR"

LOG_OUT="$TASK_DIR/logs/download_oasis3.out"
LOG_ERR="$TASK_DIR/logs/download_oasis3.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/download_oasis3_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/download_oasis3_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  OASIS-3 download + downsample (baseline rs-fMRI per subject)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  TASK_DIR:  $TASK_DIR"
echo "  OASIS3_DIR:$OASIS3_DIR/downsampled"
echo "  TMPDIR:    $TMPDIR"
echo "  Username:  $XNAT_USERNAME"
echo "============================================================"

# Install missing pkgs (idempotent)
for pkg in requests nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

EXTRA_ARGS=""
[ -n "${LIMIT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"

cd "$OFFICIAL_DIR"
python "$TASK_DIR/download_oasis3.py" \
    --username "$XNAT_USERNAME" \
    --output-dir "$OASIS3_DIR/downsampled" \
    --tmp-dir "$TMPDIR" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $OASIS3_DIR/downsampled"
echo "============================================================"
