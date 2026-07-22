#!/bin/bash
# Build montage images in a SLURM job (so it survives SSH disconnects).
# One image = GxG grid of slices of the same subject (whole-brain contact sheet).
#
# Launch:  GRID=5 PER_SUBJECT=12 DST=$LAB/brain2d_montage5 sbatch -A arieljaffe fmri2d/montages.sh
# Watch:   tail -f fmri2d/logs/montages.out
#
#SBATCH --job-name=montages
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
SRC="${SRC:-$LAB_DIR/brain2d_pool}"
DST="${DST:-$LAB_DIR/brain2d_montage5}"
GRID="${GRID:-5}"
PER_SUBJECT="${PER_SUBJECT:-12}"

export TMPDIR="$LAB_DIR/tmp"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"

echo "=== montages: src=$SRC dst=$DST grid=${GRID}x${GRID} per_subject=$PER_SUBJECT  $(date) ==="
python3 fmri2d/make_montages.py --src "$SRC" --dst "$DST" --grid "$GRID" --per-subject "$PER_SUBJECT"
echo "=== done: $(ls "$DST/HCP" | wc -l) montages  $(date) ==="
