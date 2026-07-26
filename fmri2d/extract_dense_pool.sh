#!/bin/bash
# Extract a DENSE pool of consecutive axial slices per subject (low-res, from local .pt).
# Needed for neighbor-slice local crops (z-1/z+1 must be true neighbors).
#
#   DST=$LAB/brain2d_dense sbatch -A arieljaffe fmri2d/extract_dense_pool.sh
#   tail -f fmri2d/logs/dense-pool.out
#
#SBATCH --job-name=dense-pool
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
DST="${DST:-$LAB_DIR/brain2d_dense}"
[ "${DST#/sci/}" = "$DST" ] && { echo "WARN: repinning DST under $LAB_DIR"; DST="$LAB_DIR/$(basename "$DST")"; }
SRC_GLOB="${SRC_GLOB:-$LAB_DIR/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt}"
SIZE="${SIZE:-224}"

export TMPDIR="$LAB_DIR/tmp"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official/fmri2d:${PYTHONPATH:-}"

echo "=== dense pool: src=$SRC_GLOB dst=$DST size=$SIZE  $(date) ==="
python3 fmri2d/extract_dense_pool.py --input "$SRC_GLOB" --output "$DST" --axis 2 --size "$SIZE"
echo "=== done: $(ls "$DST/HCP" 2>/dev/null | wc -l) slices in $DST/HCP  $(date) ==="
