#!/bin/bash
# =====================================================================
# Download UCLA CNP (ds000030) preprocessed rest-fMRI (fMRIPrep MNI, public
# OpenNeuro S3) -> downsample to (45,54,45) + participants.tsv. CPU-only, no login.
# Same-dataset comparison to NeuroSTORM (UCLA): schizophrenia & ADHD vs control.
#
# Usage:
#   sbatch -A arieljaffe tasks/data_prep/download_ucla/download_ucla.sh
# =====================================================================

#SBATCH --job-name=dl-ucla
#SBATCH --account=arieljaffe
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
TASK_DIR="$LAB_DIR/repos/FAIR_official/tasks/data_prep/download_ucla"
VENV_DIR="$LAB_DIR/torch_env"
OUT_DIR="${OUT_DIR:-$LAB_DIR/UCLA_data/downsampled}"
LABELS="${LABELS:-$LAB_DIR/UCLA_data/ucla_participants.tsv}"

mkdir -p "$TASK_DIR/logs"
exec >"$TASK_DIR/logs/download_ucla.out" 2>"$TASK_DIR/logs/download_ucla.err"

source "$VENV_DIR/bin/activate"

echo "UCLA CNP download   $(date)   -> $OUT_DIR"
python3 "$TASK_DIR/download_ucla.py" \
    --output-dir "$OUT_DIR" --labels-out "$LABELS" --tmp-dir "$LAB_DIR/tmp/ucla_raw"
echo "Done: $(date)"
