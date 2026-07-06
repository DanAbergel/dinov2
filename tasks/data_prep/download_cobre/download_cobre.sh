#!/bin/bash
# =====================================================================
# Download COBRE schizophrenia rs-fMRI (NIAK preprocessed, public figshare)
# -> downsample to (45,54,45) + cobre_labels.csv. CPU-only, no login/S3.
# NeuroSTORM disease benchmark: SZ vs control (146 subjects).
#
# Usage:
#   sbatch -A arieljaffe tasks/data_prep/download_cobre/download_cobre.sh
#   # (needs internet on the node; else run the python line on the login node.)
# =====================================================================

#SBATCH --job-name=dl-cobre
#SBATCH --account=arieljaffe
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
TASK_DIR="$LAB_DIR/repos/FAIR_official/tasks/data_prep/download_cobre"
VENV_DIR="$LAB_DIR/torch_env"
OUT_DIR="${OUT_DIR:-$LAB_DIR/COBRE_data/downsampled}"
LABELS="${LABELS:-$LAB_DIR/COBRE_data/cobre_labels.csv}"

mkdir -p "$TASK_DIR/logs"
exec >"$TASK_DIR/logs/download_cobre.out" 2>"$TASK_DIR/logs/download_cobre.err"

source "$VENV_DIR/bin/activate"

echo "COBRE download   $(date)   -> $OUT_DIR"
python3 "$TASK_DIR/download_cobre.py" \
    --output-dir "$OUT_DIR" --labels-out "$LABELS" --tmp-dir "$LAB_DIR/tmp/cobre_raw"
echo "Done: $(date)"
