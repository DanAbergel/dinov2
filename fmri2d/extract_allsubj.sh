#!/bin/bash
# Extract N varied-position axial slices for EVERY HCP subject -> brain2d_varied.
# Full dataset with high inter-image variety (different slice heights) so DINO's target
# is NOT uniform and dino_local can actually drop. ~1084 subjects x N slices.
# Then train+probe with ablation.sh on DATA_ROOT=brain2d_varied.
#
# Launch:  N=8 sbatch -A arieljaffe fmri2d/extract_allsubj.sh
# Watch:   tail -f fmri2d/logs/extract-allsubj.out
#
#SBATCH --job-name=extract-allsubj
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
GLOB="$LAB_DIR/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt"
MODE="${MODE:-allsubj_tp}"                     # allsubj_tp = each image a DIFFERENT timepoint AND position
                                               # allsubj    = temporal-mean volume, positions only
N="${N:-8}"                                   # images per subject
OUT="${OUT:-$LAB_DIR/brain2d_tp_varied}"

export TMPDIR="$LAB_DIR/tmp"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

echo "=== $MODE: every subject x $N images -> $OUT   $(date) ==="
python3 fmri2d/extract_slice_variety.py --mode "$MODE" --n-slices "$N" --axis 2 --size 224 \
    --input "$GLOB" --output "$OUT"
echo "=== done: $(ls "$OUT/HCP" | wc -l) PNGs   $(date) ==="
