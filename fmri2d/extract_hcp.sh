#!/bin/bash
# Batch-extract ONE axial 2D brain PNG per HCP scan -> ImageFolder (brain2d/HCP/*.png).
# Output goes OUTSIDE the repo (it's training data, not code / not committed).
# Then train standard 2D DINOv2 with dataset_path=ImageFolder:root=<OUT>.
#
# Launch:  sbatch -A arieljaffe fmri2d/extract_hcp.sh
# Watch:   tail -f fmri2d/extract_hcp.out
#
#SBATCH --job-name=fmri-extract-hcp
#SBATCH --account=arieljaffe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/extract_hcp.out
#SBATCH --error=fmri2d/extract_hcp.out
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OUT="${OUT:-$LAB_DIR/brain2d}"                 # ImageFolder root (outside the repo)
HCP_GLOB="$LAB_DIR/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt"

source "$LAB_DIR/torch_env/bin/activate"

N=$(ls $HCP_GLOB 2>/dev/null | wc -l)
echo "=== extract HCP -> $OUT/HCP   ($N scans)   $(date) ==="

# quote the glob so python (not bash) expands it
python fmri2d/extract_slices.py \
    --input "$HCP_GLOB" \
    --output "$OUT" --class-name HCP --axis 2 --size 224

echo "=== done $(date) ==="
echo "PNGs written: $(ls "$OUT/HCP" 2>/dev/null | wc -l)"
echo "ImageFolder root = $OUT   (use dataset_path=ImageFolder:root=$OUT)"
