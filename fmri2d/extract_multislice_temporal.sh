#!/bin/bash
# Multi-slice temporal pool: M spanning slices x N consecutive timepoints per subject (low-res).
# For neighbor-in-time SSL with DINO_NEIGHBOR_MODE=slice_time.
#
#   N_SLICES=10 N_TIMES=16 DST=$LAB/brain2d_ms_temporal sbatch -A arieljaffe fmri2d/extract_multislice_temporal.sh
#   tail -f fmri2d/logs/ms-temporal-pool.out
#
#SBATCH --job-name=ms-temporal-pool
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
DST="${DST:-$LAB_DIR/brain2d_ms_temporal}"
[ "${DST#/sci/}" = "$DST" ] && { echo "WARN: repinning DST under $LAB_DIR"; DST="$LAB_DIR/$(basename "$DST")"; }
SRC_GLOB="${SRC_GLOB:-$LAB_DIR/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt}"
N_SLICES="${N_SLICES:-10}"; N_TIMES="${N_TIMES:-16}"; START="${START:-100}"; SIZE="${SIZE:-224}"

export TMPDIR="$LAB_DIR/tmp"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official/fmri2d:${PYTHONPATH:-}"

echo "=== ms-temporal pool: src=$SRC_GLOB dst=$DST slices=$N_SLICES times=$N_TIMES start=$START  $(date) ==="
python3 fmri2d/extract_multislice_temporal.py --input "$SRC_GLOB" --output "$DST" \
    --n-slices "$N_SLICES" --n-times "$N_TIMES" --start "$START" --size "$SIZE"
echo "=== done: $(ls "$DST/HCP" 2>/dev/null | wc -l) images in $DST/HCP  $(date) ==="
