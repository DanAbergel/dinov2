#!/bin/bash
# =====================================================================
# SLURM Job — HCP linear probe on a DINOv2-fmri intermediate checkpoint.
#
# Same pipeline as probe_adni.sh, on HCP labels (Sex / Age / BrainVol /
# GrayMatterVol / FluidIntel / ProcSpeed / WorkingMem).
#
# HCP has T=1200 + spatial (45,54,45) = same as training -> no
# pos_temporal resize, no spatial resize.
#
# Usage:
#     sbatch slurm_jobs/probe_hcp.sh                  # auto-pick latest ckpt
#     CHECKPOINT=outputs/.../model_0005999.rank_0.pth \
#         sbatch slurm_jobs/probe_hcp.sh
# =====================================================================

#SBATCH --job-name=probe-hcp
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# SLURM's own output is discarded; we redirect to versioned files (v1, v2, ...)
# in-script below so reruns never overwrite previous logs and the filenames
# don't depend on the job id.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export FAIR_DIR="$LAB_DIR/repos/FAIR"
export VENV_DIR="$LAB_DIR/torch_env"

export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs"
mkdir -p "$OFFICIAL_DIR/outputs/probes"

# Versioned log files (v1, v2, ...) — see probe_adni.sh for the reasoning.
LOG_BASE="$OFFICIAL_DIR/slurm_jobs/logs/probe_hcp"
V=1
while [ -e "${LOG_BASE}_v${V}.out" ] || [ -e "${LOG_BASE}_v${V}.err" ]; do
    V=$((V + 1))
done
LOG_OUT="${LOG_BASE}_v${V}.out"
LOG_ERR="${LOG_BASE}_v${V}.err"
ln -sf "$(basename "$LOG_OUT")" "${LOG_BASE}_latest.out"
ln -sf "$(basename "$LOG_ERR")" "${LOG_BASE}_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

echo "============================================================"
echo "  HCP linear probe on DINOv2-fmri checkpoint"
echo "============================================================"
echo "  Job ID: ${SLURM_JOB_ID:-(local)}    Node: $(hostname)    Date: $(date)"
echo "============================================================"

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

# Ensure probe deps (idempotent).
declare -A REQUIRED_PKGS=(
    [torchmetrics]=torchmetrics
    [sklearn]=scikit-learn
    [pandas]=pandas
)
for mod in "${!REQUIRED_PKGS[@]}"; do
    if ! python -c "import $mod" 2>/dev/null; then
        echo "  $mod missing -> pip install ${REQUIRED_PKGS[$mod]}"
        pip install --no-input "${REQUIRED_PKGS[$mod]}"
    fi
done

# Pick the checkpoint (env var overrides auto).
if [ -z "${CHECKPOINT:-}" ]; then
    LATEST_RUN=$(ls -td outputs/dinov2_fmri_* 2>/dev/null | head -1 || true)
    [ -z "$LATEST_RUN" ] && { echo "ERROR: no outputs/dinov2_fmri_* runs found"; exit 2; }
    CHECKPOINT=$(ls -t "$LATEST_RUN"/model_*.rank_0.pth 2>/dev/null | head -1 || true)
    [ -z "$CHECKPOINT" ] && { echo "ERROR: no model_*.rank_0.pth in $LATEST_RUN"; exit 3; }
fi
echo "  Checkpoint: $CHECKPOINT"

# Derive iteration tag. model_<N>.rank_0.pth: use filename. model_final.rank_0.pth:
# read 'iteration' from the .pth payload.
ITER_RAW=$(basename "$CHECKPOINT" | sed -E 's/^model_0*([0-9]+)\.rank_0\.pth$/\1/')
if [[ "$ITER_RAW" =~ ^[0-9]+$ ]]; then
    ITER_TAG=$(printf "%07d" "$ITER_RAW")
else
    ITER_TAG=$(python - <<PY
import torch
try:
    d = torch.load("$CHECKPOINT", map_location="cpu", weights_only=False)
    print(f"{int(d.get('iteration', -1)):07d}")
except Exception:
    print("unknown")
PY
)
fi
OUTPUT_JSON="$OFFICIAL_DIR/outputs/probes/probe_hcp_iter${ITER_TAG}.json"
FEATURES_CACHE="$OFFICIAL_DIR/outputs/probes/features_hcp_iter${ITER_TAG}.npz"
echo "  Output:     $OUTPUT_JSON"

# Build the model from the RUN'S OWN saved config (T, temporal_kernel, ...),
# not the live fmri_vits.yaml which is mutable and may describe a different run.
CONFIG_FILE="$(dirname "$CHECKPOINT")/config.yaml"
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="dinov2/configs/train/fmri_vits.yaml"
echo "  Config:     $CONFIG_FILE"

python scripts/probe_hcp.py \
    --checkpoint "$CHECKPOINT" \
    --config-file "$CONFIG_FILE" \
    --output "$OUTPUT_JSON" \
    --features-out "$FEATURES_CACHE"

echo ""
echo "============================================================"
echo "  Done: $(date)    JSON: $OUTPUT_JSON"
echo "============================================================"
