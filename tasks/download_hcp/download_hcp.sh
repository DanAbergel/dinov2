#!/bin/bash
# =====================================================================
# SLURM job — download + downsample HCP-YA rs-fMRI from AWS S3.
#
# Streams: download 1 run from s3, downsample to (45,54,45) with
# trilinear interpolation, save .pt, delete raw. ~5 min per session.
#
# Prereqs on Moriah:
#   - AWS credentials in ~/.aws/credentials profile "hcp"
#   - python pkgs: boto3, nibabel, pandas (install once via pip)
#
# Self-contained task layout — script, sbatch and logs all live in
# tasks/download_hcp/ :
#   tasks/download_hcp/
#       download_hcp.py     (the python script)
#       download_hcp.sh     (this file)
#       logs/               (logs land here automatically)
#
# Usage:
#   sbatch tasks/download_hcp/download_hcp.sh
#   # or with custom limit / all-sessions:
#   LIMIT=50 ALL=1 sbatch tasks/download_hcp/download_hcp.sh
# =====================================================================

#SBATCH --job-name=hcp-dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/download_hcp"
export HCP_DIR="$LAB_DIR/HCP_data"
export VENV_DIR="$LAB_DIR/torch_env"
export TMPDIR="$LAB_DIR/tmp/hcp_raw"
export PYTHONUNBUFFERED=1

mkdir -p "$TASK_DIR/logs"
mkdir -p "$HCP_DIR/downsampled"
mkdir -p "$TMPDIR"

LOG_OUT="$TASK_DIR/logs/download_hcp.out"
LOG_ERR="$TASK_DIR/logs/download_hcp.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/download_hcp_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/download_hcp_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  HCP-YA download + downsample (trilinear -> 45x54x45)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  TASK_DIR:  $TASK_DIR"
echo "  HCP_DIR:   $HCP_DIR/downsampled"
echo "  TMPDIR:    $TMPDIR"
echo "============================================================"

# Install missing pkgs (idempotent)
for pkg in boto3 nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

EXTRA_ARGS=""
[ -n "${LIMIT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
[ "${ALL:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --all-sessions"

cd "$OFFICIAL_DIR"
python "$TASK_DIR/download_hcp.py" \
    --output-dir "$HCP_DIR/downsampled" \
    --tmp-dir "$TMPDIR" \
    --aws-profile hcp \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $HCP_DIR/downsampled"
echo "============================================================"
