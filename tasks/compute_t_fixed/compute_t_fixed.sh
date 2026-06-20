#!/bin/bash
# =====================================================================
# SLURM job — compute the maximal T_fixed for the mixed fMRI corpus.
#
# Reads every scan's native T, upsamples to TARGET_TR=0.72s, and reports
# the global minimum upsampled length = the largest T_fixed window that
# fits every scan with no padding (+ per-dataset minima + shortest scan).
#
# Read-only, CPU-only, ~1-2 min. Usage:
#   sbatch tasks/compute_t_fixed/compute_t_fixed.sh
#   MARGIN=2 sbatch tasks/compute_t_fixed/compute_t_fixed.sh   # safety margin
# =====================================================================

#SBATCH --job-name=compute-tfixed
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/compute_t_fixed"
VENV_DIR="$LAB_DIR/torch_env"
MARGIN="${MARGIN:-0}"

mkdir -p "$TASK_DIR/logs"
LOG_OUT="$TASK_DIR/logs/compute_t_fixed.out"
LOG_ERR="$TASK_DIR/logs/compute_t_fixed.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/compute_t_fixed_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/compute_t_fixed_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  Compute T_fixed_max   Node: $(hostname)   Date: $(date)"
echo "============================================================"

# resample_poly (imported by fmri_data) needs scipy.
python -c "import scipy" 2>/dev/null || pip install --no-input scipy

python3 "$TASK_DIR/compute_t_fixed.py" --margin "$MARGIN"

echo ""
echo "Done: $(date)"
