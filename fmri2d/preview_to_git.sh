#!/bin/bash
# Extract preview brain slices from ONE fMRI scan and PUSH them to git, so they can
# be reviewed remotely (no manual download needed).
#
# Usage (on Moriah):
#   bash fmri2d/preview_to_git.sh /path/to/scan.nii.gz      # a specific scan
#   bash fmri2d/preview_to_git.sh /path/to/corpus_dir       # auto-picks the first NIfTI
#
# Tip: for this preview prefer an HCP/ABIDE scan (permissive) rather than ADNI (DUA),
# since the PNG lands on the GitHub fork. A single 2D slice is de-identified anyway.
set -euo pipefail

ARG="${1:?usage: preview_to_git.sh <scan.nii.gz | corpus_dir>}"
cd "$(git rev-parse --show-toplevel)"

# resolve to an actual NIfTI file
if [ -d "$ARG" ]; then
    SCAN="$(find "$ARG" \( -name '*.nii.gz' -o -name '*.nii' \) | head -1)"
    echo "auto-picked scan: $SCAN"
else
    SCAN="$ARG"
fi
[ -f "$SCAN" ] || { echo "ERROR: no NIfTI found at $ARG"; exit 1; }

# activate the env that has nibabel (torch_env)
source /sci/labs/arieljaffe/dan.abergel1/torch_env/bin/activate 2>/dev/null || true

# extract all 3 planes so we can see which looks best
rm -rf fmri2d/preview
python fmri2d/extract_slices.py --input "$SCAN" --output fmri2d/preview --class-name all --all-axes --size 224

# record which scan it came from
echo "scan: $SCAN" > fmri2d/preview/SOURCE.txt

# commit + push so it shows up on the fork
git add -A fmri2d/preview
git commit -q -m "fMRI 2D preview: brain slices from $(basename "$SCAN")"
git push origin HEAD
echo "PUSHED. preview PNGs in fmri2d/preview/all/  (axis0=sagittal, axis1=coronal, axis2=axial)"
