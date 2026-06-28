#!/bin/bash
# =====================================================================
# SLURM job — inspect the multi-source fMRI corpus.
#
# Read-only. Prints per-dataset file counts, sample tensor shapes, T
# distributions, and where each dataset's native TR comes from. Output
# is what we need to wire MixedFMRIDataset (paths / shapes / TR sources).
#
# CPU-only, ~1 min. You can also just run the .py directly on the gateway:
#   source $VENV/bin/activate && python3 tasks/data_prep/inspect_corpus/inspect_corpus.py
#
# Usage:
#   sbatch tasks/data_prep/inspect_corpus/inspect_corpus.sh
# =====================================================================

#SBATCH --job-name=inspect-corpus
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/data_prep/inspect_corpus"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$TASK_DIR/logs"
LOG_OUT="$TASK_DIR/logs/inspect_corpus.out"
LOG_ERR="$TASK_DIR/logs/inspect_corpus.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/inspect_corpus_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/inspect_corpus_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo "  Inspect corpus   Node: $(hostname)   Date: $(date)"
echo "============================================================"

python3 "$TASK_DIR/inspect_corpus.py" --lab "$LAB_DIR"

echo ""
echo "Done: $(date)"
