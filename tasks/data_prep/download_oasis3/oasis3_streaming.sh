#!/bin/bash
# =====================================================================
# SLURM ARRAY job — OASIS-3 streaming download (parallel, low disk).
#
# Each array task is one shard: it processes subjects[shard_id::n_shards],
# doing download -> downsample -> delete-raw PER SUBJECT. No raw ever
# accumulates, so disk stays minimal regardless of corpus size.
#
# The number of parallel shards = the array size. With --array=0-9 you get
# 10 shards running in parallel, each handling ~1/10 of the subjects.
#
# Prereqs:
#   - nibabel installed in the venv (the gateway has internet; install once:
#       source $VENV/bin/activate && pip install nibabel)
#   - password in ~/.xnat_password (chmod 600) or env XNAT_PASSWORD
#
# Usage:
#   sbatch --array=0-9 tasks/data_prep/download_oasis3/oasis3_streaming.sh
#   # 10 parallel shards. Adjust --array=0-N for more/fewer.
# =====================================================================

#SBATCH --job-name=oasis3-stream
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/download_oasis3"
OASIS3_DIR="$LAB_DIR/OASIS3_data"
VENV_DIR="$LAB_DIR/torch_env"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-10}}"
XNAT_USERNAME="${XNAT_USERNAME:-danab95}"

# Where the downsampled .pt files go. Override to write into a shared
# lab folder, e.g. OUTPUT_DIR=/sci/labs/arieljaffe/arieljaffe/OASIS3_data/downsampled
# The per-subject structure (<subject>/rest_d<XXX>_downsampled.pt) is unchanged.
OUTPUT_DIR="${OUTPUT_DIR:-$OASIS3_DIR/downsampled}"

mkdir -p "$TASK_DIR/logs" "$OUTPUT_DIR"
TMP_DIR="$LAB_DIR/tmp/oasis3_shard${SHARD_ID}"
mkdir -p "$TMP_DIR"

LOG_OUT="$TASK_DIR/logs/oasis3_stream_shard${SHARD_ID}.out"
LOG_ERR="$TASK_DIR/logs/oasis3_stream_shard${SHARD_ID}.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  OASIS-3 streaming  shard ${SHARD_ID} / ${N_SHARDS}"
echo "  Node: $(hostname)   Date: $(date)"
echo "============================================================"

# nibabel must be preinstalled (compute nodes have no internet).
python -c "import nibabel" 2>/dev/null || {
    echo "ERROR: nibabel not installed in venv. Run on the gateway:" >&2
    echo "  source $VENV_DIR/bin/activate && pip install nibabel" >&2
    exit 4
}

echo "  OUTPUT_DIR: $OUTPUT_DIR"

XNAT_USERNAME="$XNAT_USERNAME" python "$TASK_DIR/oasis3_streaming.py" \
    --username "$XNAT_USERNAME" \
    --output-dir "$OUTPUT_DIR" \
    --tmp-dir "$TMP_DIR" \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS"

echo "Done: $(date)"
