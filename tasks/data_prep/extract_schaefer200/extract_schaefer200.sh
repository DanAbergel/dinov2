#!/bin/bash
# =====================================================================
# SLURM job — extract Schaefer-200 ROI time-series from HCP-YA volumes.
#
# For each subject_<id>/rfMRI_*.pt found under $INPUT_DIR, computes the
# mean BOLD signal per ROI per timepoint using the Schaefer-2018 atlas
# (200 ROIs, 7 Yeo networks), and saves one (T, 200) tensor per
# (subject, run).
#
# Output:
#   - Defaults to $SLURM_SUBMIT_DIR/schaefer200_output  (i.e. the cwd
#     where you ran `sbatch ...`).
#   - Override with: OUTPUT_DIR=/some/path sbatch ...
#
# Usage:
#   cd ~                                                          # or anywhere
#   sbatch /sci/.../FAIR_official/tasks/data_prep/extract_schaefer200/extract_schaefer200.sh
#   # -> outputs in ~/schaefer200_output/
#
#   # Or test on a few subjects first:
#   LIMIT=5 sbatch tasks/data_prep/extract_schaefer200/extract_schaefer200.sh
# =====================================================================

#SBATCH --job-name=schaefer200
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/extract_schaefer200"
export INPUT_DIR="${INPUT_DIR:-$LAB_DIR/HCP_data/downsampled}"
export VENV_DIR="$LAB_DIR/torch_env"
export PYTHONUNBUFFERED=1

# Output dir = where the user submitted the sbatch from (cwd at submission).
# SLURM_SUBMIT_DIR is set by SLURM. Falls back to $PWD if running locally.
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/schaefer200_output}"

mkdir -p "$TASK_DIR/logs"
mkdir -p "$OUTPUT_DIR"

LOG_OUT="$TASK_DIR/logs/extract_schaefer200.out"
LOG_ERR="$TASK_DIR/logs/extract_schaefer200.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/extract_schaefer200_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/extract_schaefer200_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  Extract Schaefer-200 ROI time-series"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  INPUT_DIR:  $INPUT_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "============================================================"

# Install missing pkgs (idempotent)
for pkg in nilearn nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

EXTRA_ARGS=""
[ -n "${LIMIT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"

cd "$OFFICIAL_DIR"
python "$TASK_DIR/extract_schaefer200.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
