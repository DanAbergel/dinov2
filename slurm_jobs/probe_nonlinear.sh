#!/bin/bash
# =====================================================================
# ADNI non-linear probe: compares a LINEAR head vs a small MLP head on the
# same frozen CLS features. Tests whether non-linearity unlocks more
# clinical signal than the (plateaued) linear probe.
#
# 1 GPU only to extract CLS features once (cached to --features-out so you
# can re-run the sklearn comparison on the gateway with --features-in).
#
# Usage:
#   CHECKPOINT=outputs/dinov2_fmri_20260524_144327/model_0008999.rank_0.pth \
#       sbatch slurm_jobs/probe_nonlinear.sh
# =====================================================================

#SBATCH --job-name=probe-nl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export FAIR_DIR="$LAB_DIR/repos/FAIR"
export VENV_DIR="$LAB_DIR/torch_env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs" "$OFFICIAL_DIR/outputs/probes"

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

# ----- Determine names BEFORE redirecting -----
: "${CHECKPOINT:?Set CHECKPOINT=outputs/.../model_*.rank_0.pth}"
ITER_RAW=$(basename "$CHECKPOINT" | sed -E 's/^model_0*([0-9]+)\.rank_0\.pth$/\1/')
if [[ "$ITER_RAW" =~ ^[0-9]+$ ]]; then
    TAG=$(printf "%07d" "$ITER_RAW")
else
    TAG=$(python - <<PY 2>/dev/null
import torch
try:
    d = torch.load("$CHECKPOINT", map_location="cpu", weights_only=False)
    print(f"{int(d.get('iteration', -1)):07d}")
except Exception:
    print("unknown")
PY
)
fi
RUN_DIR=$(dirname "$CHECKPOINT")
RUN_NAME=$(basename "$RUN_DIR")
CONFIG_FILE="$RUN_DIR/config.yaml"
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="dinov2/configs/train/fmri_vits.yaml"
FEATURES="$OFFICIAL_DIR/outputs/probes/features_nonlinear_${RUN_NAME}_iter${TAG}.npz"
OUTPUT="$OFFICIAL_DIR/outputs/probes/probe_nonlinear_${RUN_NAME}_iter${TAG}.json"

# ----- Self-identifying log file -----
LOG_OUT="$OFFICIAL_DIR/slurm_jobs/logs/probe_nonlinear_${RUN_NAME}_iter${TAG}.out"
LOG_ERR="$OFFICIAL_DIR/slurm_jobs/logs/probe_nonlinear_${RUN_NAME}_iter${TAG}.err"
ln -sf "$(basename "$LOG_OUT")" "$OFFICIAL_DIR/slurm_jobs/logs/probe_nonlinear_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$OFFICIAL_DIR/slurm_jobs/logs/probe_nonlinear_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

echo "============================================================"
echo "  ADNI non-linear probe (linear vs MLP)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  RUN_NAME:   $RUN_NAME"
echo "  Iteration:  $TAG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config:     $CONFIG_FILE"
echo "  Output:     $OUTPUT"
echo "============================================================"

for pkg in scikit-learn; do
    mod=$(echo "$pkg" | tr - _)
    python -c "import $mod" 2>/dev/null || pip install --no-input "$pkg"
done

python scripts/probe_adni_nonlinear.py \
    --checkpoint "$CHECKPOINT" \
    --config-file "$CONFIG_FILE" \
    --features-out "$FEATURES" \
    --output "$OUTPUT" \
    --n_repeats 4

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  JSON:     $OUTPUT"
echo "  Features: $FEATURES  (re-run on gateway with --features-in)"
echo "============================================================"
