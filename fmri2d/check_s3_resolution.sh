#!/bin/bash
# Diagnostic: download ONE HCP fMRI volume from S3 (the original) and compare its spatial
# resolution to OUR local .pt version, to see if ours is downsampled.
#
# Needs: awscli + HCP AWS credentials (from ConnectomeDB, after DUA) in the environment
#        (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials), and nibabel (in torch_env).
#
# Launch:  SUBJ=100206 sbatch -A arieljaffe fmri2d/check_s3_resolution.sh
# Watch:   tail -f fmri2d/logs/check-s3.out
#
#SBATCH --job-name=check-s3
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
SUBJ="${SUBJ:-100206}"
OUR_PT="${OUR_PT:-$LAB_DIR/HCP_data/downsampled/subject_$SUBJ/rfMRI_REST1_LR_downsampled.pt}"
S3_PATH="${S3_PATH:-s3://hcp-openaccess/HCP_1200/$SUBJ/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz}"

export TMPDIR="$LAB_DIR/tmp"; export HOME="$LAB_DIR"; export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p fmri2d/logs "$TMPDIR"
source "$LAB_DIR/torch_env/bin/activate"

command -v aws >/dev/null || { echo "ERROR: awscli not found -> pip install awscli, or module load awscli"; exit 1; }
[ -f "$OUR_PT" ] || { echo "ERROR: our local volume not found: $OUR_PT"; exit 1; }

TMP="$TMPDIR/hcp_${SUBJ}_orig.nii.gz"
echo "=== downloading original from S3: $S3_PATH  $(date) ==="
aws s3 cp "$S3_PATH" "$TMP"

echo "=== comparing resolutions ==="
python3 - "$OUR_PT" "$TMP" <<'EOF'
import sys, numpy as np, torch, nibabel as nib

our_pt, s3_nii = sys.argv[1], sys.argv[2]

ours = np.squeeze(torch.load(our_pt, map_location="cpu", weights_only=True).numpy())
img = nib.load(s3_nii)
orig = np.squeeze(img.get_fdata())

# our .pt is (T,X,Y,Z) [T first]; NIfTI is (X,Y,Z,T) [T last] -> compare the 3 spatial dims
our_spatial = tuple(sorted(ours.shape)[-3:]) if ours.ndim == 4 else tuple(ours.shape)
orig_spatial = tuple(img.shape[:3])
vox = img.header.get_zooms()[:3]

print(f"OUR  .pt full shape : {ours.shape}")
print(f"OUR  spatial dims   : {our_spatial}")
print(f"S3   full shape     : {img.shape}")
print(f"S3   spatial dims   : {orig_spatial}   voxel size (mm): {tuple(round(float(v),3) for v in vox)}")

ov = np.prod(our_spatial); sv = np.prod(orig_spatial)
ratio = sv / max(ov, 1)
print(f"\nvoxel-count ratio (S3 / ours) = {ratio:.1f}x")
if ratio > 1.3:
    print(">>> OUR VERSION IS DOWNSAMPLED -> S3 is more detailed. Worth streaming full-res per subject.")
elif ratio < 0.77:
    print(">>> Ours is HIGHER res than this S3 file (unexpected) - check the S3 path.")
else:
    print(">>> Same resolution -> no benefit from re-downloading.")
EOF

rm -f "$TMP"
echo "=== done  $(date) ==="
