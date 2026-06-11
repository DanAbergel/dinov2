#!/bin/bash
# =====================================================================
# Diagnose a DINOv2 training run from its training_metrics.json.
#
# CPU-only (just reads JSONL), ~10 seconds.
#
# Usage:
#   RUN=dinov2_fmri_hcp_baseline sbatch slurm_jobs/diagnose_training.sh
#   # or run on all 3 in sequence:
#   bash slurm_jobs/diagnose_training.sh local
# =====================================================================

#SBATCH --job-name=diag-train
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export VENV_DIR="$LAB_DIR/torch_env"
export PYTHONUNBUFFERED=1

# Allow running outside SLURM via: bash slurm_jobs/diagnose_training.sh local
if [ "${1:-}" = "local" ]; then
    cd "$OFFICIAL_DIR"
    source "$VENV_DIR/bin/activate"
    for run in dinov2_fmri_hcp_baseline dinov2_fmri_hcp_freeze_last3 dinov2_fmri_hcp_freeze_fmri; do
        echo ""
        echo "############################################################"
        echo "##  $run"
        echo "############################################################"
        python scripts/diagnose_training.py "outputs/$run"
    done
    exit 0
fi

mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs"

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

: "${RUN:?Set RUN=dinov2_fmri_hcp_baseline (the run dir under outputs/)}"

LOG_OUT="$OFFICIAL_DIR/slurm_jobs/logs/diagnose_training_${RUN}.out"
LOG_ERR="$OFFICIAL_DIR/slurm_jobs/logs/diagnose_training_${RUN}.err"
ln -sf "$(basename "$LOG_OUT")" "$OFFICIAL_DIR/slurm_jobs/logs/diagnose_training_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$OFFICIAL_DIR/slurm_jobs/logs/diagnose_training_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

echo "============================================================"
echo "  Diagnose DINOv2 training: $RUN"
echo "============================================================"
echo "  Job ID : ${SLURM_JOB_ID:-(local)}"
echo "  Node   : $(hostname)"
echo "  Date   : $(date)"
echo "============================================================"

python scripts/diagnose_training.py "outputs/$RUN"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
