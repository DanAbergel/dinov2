#!/bin/bash
# =====================================================================
# SLURM job — download + downsample HCP-YA TASK-fMRI from AWS S3.
# NeuroSTORM task-state benchmark: 7 tasks (EMOTION/GAMBLING/LANGUAGE/
# MOTOR/RELATIONAL/SOCIAL/WM), label = which task -> 7-class classification.
#
# Same streaming as download_hcp (rest): download 1 run, downsample to
# (45,54,45), save .pt, delete raw. Writes a manifest CSV with the task label.
#
# Prereqs on Moriah:
#   - FRESH ConnectomeDB S3 keys in ~/.aws/credentials (the old ones expire ->
#     InvalidAccessKeyId). Refresh at db.humanconnectome.org -> Amazon S3 Access.
#   - boto3, nibabel (auto-installed if missing)
#
# Usage:
#   # confirm S3 paths on 2 subjects first (cheap):
#   LIMIT=2 sbatch tasks/data_prep/download_hcp_task/download_hcp_task.sh
#   # full run (7 tasks x LR per subject); ALLPE=1 for LR+RL:
#   sbatch tasks/data_prep/download_hcp_task/download_hcp_task.sh
# =====================================================================

#SBATCH --job-name=hcp-task-dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/download_hcp_task"
export HCP_DIR="$LAB_DIR/HCP_task_data"
export VENV_DIR="$LAB_DIR/torch_env"
export TMPDIR="$LAB_DIR/tmp/hcp_task_raw"
export PYTHONUNBUFFERED=1

mkdir -p "$TASK_DIR/logs" "$HCP_DIR/downsampled" "$TMPDIR"

LOG_OUT="$TASK_DIR/logs/download_hcp_task.out"
LOG_ERR="$TASK_DIR/logs/download_hcp_task.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  HCP-YA TASK-fMRI download + downsample (-> 45x54x45)"
echo "  Job ${SLURM_JOB_ID:-(local)}  Node $(hostname)  $(date)"
echo "  out: $HCP_DIR/downsampled"
echo "============================================================"

for pkg in boto3 nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

EXTRA_ARGS=""
[ -n "${LIMIT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
[ "${ALLPE:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --all-pe"
[ -n "${TASKS:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --tasks $TASKS"

cd "$OFFICIAL_DIR"
python "$TASK_DIR/download_hcp_task.py" \
    --output-dir "$HCP_DIR/downsampled" \
    --manifest "$HCP_DIR/hcp_task_labels.csv" \
    --tmp-dir "$TMPDIR" \
    --aws-profile "${AWS_PROFILE:-default}" \
    $EXTRA_ARGS

echo ""
echo "  Done: $(date)   manifest: $HCP_DIR/hcp_task_labels.csv"
