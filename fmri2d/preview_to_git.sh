#!/bin/bash
# Extract preview brain slices from ONE fMRI scan and push them to git, so they can
# be reviewed remotely. Runs as a SLURM job (the full ~1200-frame .pt load needs RAM).
#
# Usage (on Moriah):
#   sbatch -A arieljaffe fmri2d/preview_to_git.sh /sci/labs/arieljaffe/dan.abergel1/HCP_data/downsampled
#   sbatch -A arieljaffe fmri2d/preview_to_git.sh /path/to/one_scan.pt
# Watch:  tail -f fmri2d/preview_job.out
#
#SBATCH --job-name=fmri-preview
#SBATCH --account=arieljaffe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/preview_job.out
#SBATCH --error=fmri2d/preview_job.err
set -euo pipefail

echo "=== fmri-preview: start $(date) ==="
ARG="${1:?usage: sbatch -A arieljaffe fmri2d/preview_to_git.sh <scan.pt | corpus_dir>}"
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"

# resolve to an actual scan file (.pt or NIfTI). Use `-print -quit` (find stops itself
# at the first match) — NOT `| head -1`, which SIGPIPE-kills find under `set -o pipefail`.
if [ -d "$ARG" ]; then
    SCAN="$(find "$ARG" \( -name '*.pt' -o -name '*.nii.gz' -o -name '*.nii' \) -print -quit)"
    echo "auto-picked scan: $SCAN"
else
    SCAN="$ARG"
fi
[ -f "$SCAN" ] || { echo "ERROR: no scan (.pt/.nii) found at $ARG"; exit 1; }

source "$LAB_DIR/torch_env/bin/activate"

# extract all 3 planes (full temporal mean)
rm -rf fmri2d/preview
python fmri2d/extract_slices.py --input "$SCAN" --output fmri2d/preview --class-name all --all-axes --size 224
echo "scan: $SCAN" > fmri2d/preview/SOURCE.txt
ls -la fmri2d/preview/all/

# commit + push (push may fail from a compute node if git creds aren't available there —
# the commit is still made, just run `git push` from the login node afterwards).
git add -A fmri2d/preview
git commit -q -m "fMRI 2D preview: brain slices from $(basename "$SCAN")" || echo "nothing to commit"
if git push origin HEAD; then
    echo "PUSHED. preview PNGs in fmri2d/preview/all/  (axis0=sagittal, axis1=coronal, axis2=axial)"
else
    echo "COMMIT DONE but PUSH FAILED from compute node -> run 'git push' on the login node."
fi
