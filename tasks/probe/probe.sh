#!/bin/bash
# =====================================================================
# Leakage-free linear probe on ADNI for one trained run.
#
# Loads the run's teacher encoder, extracts ADNI embeddings, fits LogReg on
# TRAIN / selects C on VAL / reports TEST AUC (subjects from subject_split.json;
# val+test were held out of pretraining). Writes <run-dir>/probe_adni.json.
#
# Usage:
#   RUN=fmri_v2_baseline sbatch -A arieljaffe tasks/probe/probe.sh
#   RUN=fmri_v2_fourier  CKPT=model_final.rank_0.pth sbatch -A arieljaffe tasks/probe/probe.sh
# =====================================================================

#SBATCH --job-name=probe
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/probe"
VENV_DIR="$LAB_DIR/torch_env"

RUN="${RUN:-fmri_v2_baseline}"
CKPT="${CKPT:-model_final.rank_0.pth}"
RUN_DIR="$LAB_DIR/runs/$RUN"

mkdir -p "$TASK_DIR/logs"
LOG="$TASK_DIR/logs/probe_${RUN}.out"
exec >"$LOG" 2>&1

export TMPDIR="$LAB_DIR/tmp"
export HOME="$LAB_DIR"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

echo "Probe   run=$RUN ckpt=$CKPT   Node: $(hostname)   Date: $(date)"
srun python tasks/probe/probe.py --run-dir "$RUN_DIR" --checkpoint "$CKPT"
echo "Done: $(date)"
