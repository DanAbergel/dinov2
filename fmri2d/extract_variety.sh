#!/bin/bash
# Extract the two varied-slice datasets in a SLURM job (loading a 4D fMRI volume OOM-kills
# the login node). Produces brain2d_1subj (1 subject, 16 positions) and brain2d_16subj
# (16 subjects, 16 positions). No GPU needed for the work, but --gres guarantees a slot.
#
# Launch:  sbatch -A arieljaffe fmri2d/extract_variety.sh
# Watch:   tail -f fmri2d/logs/extract-variety.out
#
#SBATCH --job-name=extract-variety
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
GLOB="$LAB_DIR/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt"
N="${N:-16}"

export TMPDIR="$LAB_DIR/tmp"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

echo "=== A) onesubj: 1 subject, $N slice positions   $(date) ==="
python3 fmri2d/extract_slice_variety.py --mode onesubj --n-slices "$N" --axis 2 --size 224 \
    --input "$GLOB" --output "$LAB_DIR/brain2d_1subj"

echo "=== B) multisubj: $N subjects, one position each   $(date) ==="
python3 fmri2d/extract_slice_variety.py --mode multisubj --n-slices "$N" --axis 2 --size 224 \
    --input "$GLOB" --output "$LAB_DIR/brain2d_16subj"

echo "=== done: $(ls "$LAB_DIR/brain2d_1subj/HCP" | wc -l) + $(ls "$LAB_DIR/brain2d_16subj/HCP" | wc -l) PNGs  $(date) ==="
