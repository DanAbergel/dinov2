#!/bin/bash
# =====================================================================
# Build the subject-level train/val/test split (70/15/15) for the corpus.
#
# Reads  <lab>/corpus_manifest.csv, writes <lab>/subject_split.json.
# MixedFMRIDataset then excludes the val+test subjects of the downstream
# datasets (ADNI/ABIDE/OASIS) from SSL pretraining (no leakage); probes
# evaluate on the held-out test subjects.
#
# Read-only-ish, CPU, <1 min. Run on the gateway or:
#   sbatch -A arieljaffe tasks/make_subject_split/make_subject_split.sh
# =====================================================================

#SBATCH --job-name=make-split
#SBATCH --account=arieljaffe
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/make_subject_split"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$TASK_DIR/logs"
LOG_OUT="$TASK_DIR/logs/make_subject_split.out"
exec >"$LOG_OUT" 2>&1

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

echo "Building subject split   $(date)"
python3 "$TASK_DIR/make_subject_split.py" --lab "$LAB_DIR"
echo "Done: $(date)"
