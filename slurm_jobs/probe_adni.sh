#!/bin/bash
# =====================================================================
# SLURM Job — ADNI linear probe on a DINOv2-fmri intermediate checkpoint.
#
# Loads any model_<iter>.rank_0.pth saved by PeriodicCheckpointer
# (every 3000 iters during run_dinov2_fmri.sh), extracts the TEACHER CLS
# embedding per ADNI scan and runs StratifiedGroupKFold k=5 with LBFGS
# LogReg / Ridge — same pipeline as your prior `dino_probe_adni_*.json`
# files for direct apples-to-apples comparison.
#
# Usage:
#     # Auto-pick the latest checkpoint of the most recent run:
#     sbatch slurm_jobs/probe_adni.sh
#
#     # Explicit checkpoint:
#     CHECKPOINT=outputs/dinov2_fmri_20260519_201234/model_0008999.rank_0.pth \
#         sbatch slurm_jobs/probe_adni.sh
#
# This job is CPU+1 GPU; it does NOT touch the training run so you can
# launch it while the training is still going.
# =====================================================================

#SBATCH --job-name=probe-adni
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_jobs/logs/probe_adni.out
#SBATCH --error=slurm_jobs/logs/probe_adni.err
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export FAIR_DIR="$LAB_DIR/repos/FAIR"           # for label loader imports in probe_adni.py
export VENV_DIR="$LAB_DIR/torch_env"

export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs"
mkdir -p "$OFFICIAL_DIR/outputs/probes"

echo "============================================================"
echo "  ADNI linear probe on DINOv2-fmri checkpoint"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "============================================================"

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

# ----- 0. Install missing official deps needed by dinov2.eval.* (idempotent) -----
# `dinov2.eval.utils` imports `torchmetrics.MetricCollection`. The train job
# skips it (training only needs fvcore for PeriodicCheckpointer), but the
# probe pulls in the eval package which needs torchmetrics.
declare -A REQUIRED_PKGS=(
    [torchmetrics]=torchmetrics
    [scikit-learn]=scikit-learn
)
for mod in "${!REQUIRED_PKGS[@]}"; do
    python_name=$(echo "$mod" | tr - _)
    if ! python -c "import $python_name" 2>/dev/null; then
        echo "  $mod missing -> pip install ${REQUIRED_PKGS[$mod]}"
        pip install --no-input "${REQUIRED_PKGS[$mod]}"
    fi
done
python -c "import torchmetrics, sklearn; print(f'  torchmetrics {torchmetrics.__version__}  sklearn {sklearn.__version__}')"

# ----- 1. Find the checkpoint (env var overrides auto-discovery) -----
if [ -z "${CHECKPOINT:-}" ]; then
    # Pick the latest model_*.rank_0.pth from the most recent run.
    LATEST_RUN=$(ls -td outputs/dinov2_fmri_* 2>/dev/null | head -1 || true)
    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: no outputs/dinov2_fmri_* runs found and no CHECKPOINT set."
        exit 2
    fi
    CHECKPOINT=$(ls -t "$LATEST_RUN"/model_*.rank_0.pth 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: no model_*.rank_0.pth in $LATEST_RUN. Wait for iter 3000+."
        exit 3
    fi
fi
echo "  Checkpoint: $CHECKPOINT"

# Derive iteration tag from filename (model_0011999.rank_0.pth -> 0011999).
ITER_TAG=$(basename "$CHECKPOINT" | sed -E 's/^model_([0-9]+)\.rank_0\.pth$/\1/')
OUTPUT_JSON="$OFFICIAL_DIR/outputs/probes/probe_iter${ITER_TAG}.json"
FEATURES_CACHE="$OFFICIAL_DIR/outputs/probes/features_iter${ITER_TAG}.npz"
echo "  Output:     $OUTPUT_JSON"

# ----- 2. Run -----
python scripts/probe_adni.py \
    --checkpoint "$CHECKPOINT" \
    --config-file dinov2/configs/train/fmri_vits.yaml \
    --output "$OUTPUT_JSON" \
    --features-out "$FEATURES_CACHE"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  JSON: $OUTPUT_JSON"
echo "============================================================"
