#!/bin/bash
# =====================================================================
# Download ADHD-200 preprocessed rs-fMRI (CPAC, fcp-indi public S3, no login)
# -> downsample to (45,54,45) + merged phenotypic CSV. CPU-only.
#
# Usage:
#   sbatch -A arieljaffe tasks/data_prep/download_adhd200/download_adhd200.sh
#   # (needs internet on the node; if the compute node has none, run the python
#   #  line directly on the login node moriah-gw.)
# =====================================================================

#SBATCH --job-name=dl-adhd200
#SBATCH --account=arieljaffe
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
TASK_DIR="$LAB_DIR/repos/FAIR_official/tasks/data_prep/download_adhd200"
VENV_DIR="$LAB_DIR/torch_env"
OUT_DIR="${OUT_DIR:-$LAB_DIR/ADHD200_data/downsampled}"
PHENO="${PHENO:-$LAB_DIR/ADHD200_data/adhd200_phenotypic.csv}"

mkdir -p "$TASK_DIR/logs"
exec >"$TASK_DIR/logs/download_adhd200.out" 2>"$TASK_DIR/logs/download_adhd200.err"

source "$VENV_DIR/bin/activate"
pip install -q boto3 2>/dev/null || true

echo "ADHD-200 download   $(date)   -> $OUT_DIR"
python3 "$TASK_DIR/download_adhd200.py" \
    --output-dir "$OUT_DIR" --pheno-out "$PHENO" --tmp-dir "$LAB_DIR/tmp/adhd_raw"
echo "Done: $(date)"
