#!/bin/bash
# Stream full-res HCP rfMRI from S3, extract PNG slices, discard each volume.
# Produces an ImageFolder at 2x the detail of our downsampled brain2d.
#
# Quick test (20 subjects, ~20 min):
#   LIMIT=20 DST=$LAB/brain2d_hires_test sbatch -A arieljaffe fmri2d/extract_s3_highres.sh
# Full run, single job (~20h):
#   DST=$LAB/brain2d_hires sbatch -A arieljaffe fmri2d/extract_s3_highres.sh
# Full run, faster via a 10-way job array (each shard ~2-3h):
#   DST=$LAB/brain2d_hires sbatch -A arieljaffe --array=0-9 fmri2d/extract_s3_highres.sh
#
#SBATCH --job-name=s3-hires
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x_%a.out
#SBATCH --error=fmri2d/logs/%x_%a.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
DST="${DST:-$LAB_DIR/brain2d_hires}"
SUBJECTS_DIR="${SUBJECTS_DIR:-$LAB_DIR/HCP_data/downsampled}"
N_FRAMES="${N_FRAMES:-0}"          # 0 = one temporal-mean image/subject (like brain2d); >0 = multi-frame
AXIS="${AXIS:-2}"
SIZE="${SIZE:-224}"
LIMIT="${LIMIT:-0}"                 # >0 = only first N subjects (quick test)
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

# AWS credentials live in the REAL home; we override HOME below (for caches), so point AWS at them
REAL_HOME="${HOME}"
export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-$REAL_HOME/.aws/credentials}"
export AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-$REAL_HOME/.aws/config}"
export TMPDIR="$LAB_DIR/tmp/s3_hires_$SHARD"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official/fmri2d:${PYTHONPATH:-}"   # so extract_s3_highres imports extract_slices

command -v aws >/dev/null || { echo "ERROR: awscli not found -> pip install awscli"; exit 1; }
echo "=== s3-hires shard $SHARD/$NSHARDS -> $DST  n_frames=$N_FRAMES limit=$LIMIT  $(date) ==="

python3 fmri2d/extract_s3_highres.py \
    --subjects-dir "$SUBJECTS_DIR" --out "$DST" --tmp "$TMPDIR" \
    --axis "$AXIS" --size "$SIZE" --n-frames "$N_FRAMES" --limit "$LIMIT" \
    --shard "$SHARD" --n-shards "$NSHARDS"

echo "=== shard $SHARD done: $(ls "$DST/HCP" 2>/dev/null | wc -l) PNGs total in $DST/HCP  $(date) ==="
rmdir "$TMPDIR" 2>/dev/null || true
