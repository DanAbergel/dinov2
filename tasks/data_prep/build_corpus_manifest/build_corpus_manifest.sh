#!/bin/bash
# =====================================================================
# SLURM job — build the corpus manifest once.
#
# Writes <lab>/corpus_manifest.csv (dataset,path,subject_id,tr,T_native,
# upsampled_T) by scanning every scan's header T. ~9 min (network FS).
# Run once; MixedFMRIDataset then reads it instead of re-scanning, and
# uses upsampled_T to drop too-short scans for the chosen T_fixed.
#
# Usage:
#   sbatch tasks/data_prep/build_corpus_manifest/build_corpus_manifest.sh
# =====================================================================

#SBATCH --job-name=corpus-manifest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/build_corpus_manifest"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$TASK_DIR/logs"
LOG_OUT="$TASK_DIR/logs/build_corpus_manifest.out"
LOG_ERR="$TASK_DIR/logs/build_corpus_manifest.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/build_corpus_manifest_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/build_corpus_manifest_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"
# dinov2 package importable (running a script file, not from repo root).
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"
python -c "import scipy" 2>/dev/null || pip install --no-input scipy

echo "============================================================"
echo "  Build corpus manifest   Node: $(hostname)   Date: $(date)"
echo "============================================================"

python3 "$TASK_DIR/build_corpus_manifest.py" --lab "$LAB_DIR"

echo ""
echo "Done: $(date)"
