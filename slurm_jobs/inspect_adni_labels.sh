#!/bin/bash
# =====================================================================
# Inspect ADNI label distributions:
#   - All distinct values of CDR / Degradation1Y / 2Y / 3Y / Sex / etc.
#   - Count and percentage of each value
#   - Histogram for continuous columns (Age, MMSE)
#
# CPU-only, no GPU needed. ~30 seconds.
#
# Usage:
#     sbatch slurm_jobs/inspect_adni_labels.sh
# =====================================================================

#SBATCH --job-name=inspect-adni
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export VENV_DIR="$LAB_DIR/torch_env"
export PYTHONUNBUFFERED=1

mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs"

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

LOG_OUT="$OFFICIAL_DIR/slurm_jobs/logs/inspect_adni_labels.out"
LOG_ERR="$OFFICIAL_DIR/slurm_jobs/logs/inspect_adni_labels.err"
ln -sf "$(basename "$LOG_OUT")" "$OFFICIAL_DIR/slurm_jobs/logs/inspect_adni_labels_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$OFFICIAL_DIR/slurm_jobs/logs/inspect_adni_labels_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

echo "============================================================"
echo "  Inspect ADNI labels"
echo "============================================================"
echo "  Job ID : ${SLURM_JOB_ID:-(local)}"
echo "  Node   : $(hostname)"
echo "  Date   : $(date)"
echo "  Out    : $LOG_OUT"
echo "============================================================"

python scripts/inspect_adni_labels.py

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
