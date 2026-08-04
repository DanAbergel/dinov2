#!/bin/bash
# Random-backbone baseline for the fMRI probe (HCP Sex, official linear probe).
# The mandatory reference for interpreting the trained probe numbers.
#   sbatch -A arieljaffe tasks/v2/probe/random_baseline.sh
#   tail -f tasks/v2/probe/logs/random_baseline.out
#SBATCH --job-name=fmri-randbase
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
CONFIG="$OFFICIAL_DIR/dinov2/configs/train/fmri_vits.yaml"
VENV="${VENV:-$LAB_DIR/torch_env}"

mkdir -p "$OFFICIAL_DIR/tasks/v2/probe/logs"
exec >"$OFFICIAL_DIR/tasks/v2/probe/logs/random_baseline.out" 2>"$OFFICIAL_DIR/tasks/v2/probe/logs/random_baseline.err"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

echo "=== RANDOM baseline  Node: $(hostname)  $(date) ==="
srun python tasks/v2/probe/random_baseline.py \
    --config-file "$CONFIG" \
    --output-dir "$LAB_DIR/runs/v2/random_baseline"
echo "=== done  $(date) ==="
