#!/bin/bash
# =====================================================================
# Diagnostic: verify that ADNIFullScanDataset._load returns the
# expected (140, 1, 45, 54, 45) shape AFTER the trilinear resample fix.
#
# CPU-only — no GPU needed, just data loading.
# Run with:  sbatch slurm_jobs/diagnose_adni_shape.sh
# Log:       slurm_jobs/logs/diagnose_adni_shape.out
# =====================================================================

#SBATCH --job-name=diag-adni-shape
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_jobs/logs/diagnose_adni_shape.out
#SBATCH --error=slurm_jobs/logs/diagnose_adni_shape.err
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export VENV_DIR="$LAB_DIR/torch_env"

mkdir -p slurm_jobs/logs

echo "============================================================"
echo "  Diagnostic ADNI shape (after trilinear resample fix)"
echo "============================================================"
echo "  Job ID:  ${SLURM_JOB_ID:-(local)}"
echo "  Node:    $(hostname)"
echo "  Date:    $(date)"
echo "  Branch:  $(git branch --show-current)"
echo "  HEAD:    $(git log --oneline -1)"
echo "============================================================"

source "$VENV_DIR/bin/activate"

echo ""
echo "[1] Module file Python actually imports:"
python3 -c "import dinov2.data.fmri_data as m; print('   LOADED FROM:', m.__file__)"

echo ""
echo "[2] Grep target_shape in the file:"
grep -n "target_shape\|F.interpolate" dinov2/data/fmri_data.py | head -20 | sed 's/^/   /'

echo ""
echo "[3] Instantiate ADNI and call _load(0):"
python3 -c "
from dinov2.data.fmri_data import ADNIFullScanDataset
adni = ADNIFullScanDataset()
out = adni._load(0)
print('   ADNI raw tensor shape:', tuple(adni.data.shape))
print('   ADNI target_shape:    ', adni.target_shape)
print('   ADNI _load(0).shape:  ', tuple(out.shape))
"

echo ""
echo "[4] Instantiate MixedFMRIDataset and grab one ADNI sample:"
python3 -c "
from dinov2.data.fmri_data import MixedFMRIDataset
mixed = MixedFMRIDataset()
adni_idx = len(mixed.hcp)
print('   total scans:', len(mixed))
print('   HCP count:  ', len(mixed.hcp))
print('   ADNI count: ', len(mixed.adni))
img, _ = mixed[adni_idx]
print('   Mixed[ADNI sample 0][0].shape =', tuple(img.shape))
"

echo ""
echo "[5] Instantiate MixedFMRIDataset WITH MultiCrop3D (real training path):"
python3 -c "
from dinov2.data.fmri_data import MixedFMRIDataset, MultiCrop3D
mc = MultiCrop3D(
    global_crops_scale=(0.32, 1.0),
    local_crops_scale=(0.05, 0.32),
    local_crops_number=8,
)
mixed = MixedFMRIDataset(transform=mc)
adni_idx = len(mixed.hcp)
img_dict, _ = mixed[adni_idx]
print('   ADNI sample dict keys:', list(img_dict.keys()))
print('   global_crops[0].shape:', tuple(img_dict['global_crops'][0].shape))
print('   global_crops[1].shape:', tuple(img_dict['global_crops'][1].shape))
print('   local_crops[0].shape: ', tuple(img_dict['local_crops'][0].shape))

# Same for first HCP sample
img_dict, _ = mixed[0]
print('   HCP global_crops[0].shape:', tuple(img_dict['global_crops'][0].shape))
"

echo ""
echo "============================================================"
echo "  Expected: every crop shape ends in (45, 54, 45)"
echo "  If ADNI crops are (46, 55, 46) -> resample fix is NOT firing"
echo "============================================================"
