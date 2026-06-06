#!/bin/bash
# =====================================================================
# SLURM Job — ADNI linear probe on a DINOv2-fmri checkpoint.
#
# Loads any model_<iter>.rank_0.pth, builds the model from THAT RUN'S
# saved config.yaml, extracts the teacher CLS embedding per ADNI scan,
# and runs StratifiedGroupKFold linear probes.
#
# Output names are SELF-IDENTIFYING (no v1/v2 versioning):
#   slurm_jobs/logs/probe_adni_<RUN_NAME>_iter<ITER>.out
#   outputs/probes/probe_adni_<RUN_NAME>_iter<ITER>.json
# Re-probing the SAME checkpoint overwrites those files. Probing a
# different checkpoint never collides. `probe_adni_latest.{out,err}`
# symlinks point at the most recent invocation for `tail -f`.
#
# Usage:
#   sbatch slurm_jobs/probe_adni.sh                       # auto-pick latest run
#   CHECKPOINT=outputs/dinov2_fmri_.../model_*.rank_0.pth \
#       sbatch slurm_jobs/probe_adni.sh                   # explicit
# =====================================================================

#SBATCH --job-name=probe-adni
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# SLURM's own output is /dev/null; we redirect to an explicit, self-named
# log file in-script once we know the checkpoint's (RUN_NAME, iter).
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

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

# ----- 1. Resolve CHECKPOINT (env var overrides auto-discovery) -----
if [ -z "${CHECKPOINT:-}" ]; then
    LATEST_RUN=$(ls -td outputs/dinov2_fmri_* 2>/dev/null | head -1 || true)
    [ -z "$LATEST_RUN" ] && exit 2
    CHECKPOINT=$(ls -t "$LATEST_RUN"/model_*.rank_0.pth 2>/dev/null | head -1 || true)
    [ -z "$CHECKPOINT" ] && exit 3
fi

# ----- 2. Derive ITER_TAG and RUN_NAME (silent — runs before log redirect) -----
ITER_RAW=$(basename "$CHECKPOINT" | sed -E 's/^model_0*([0-9]+)\.rank_0\.pth$/\1/')
if [[ "$ITER_RAW" =~ ^[0-9]+$ ]]; then
    ITER_TAG=$(printf "%07d" "$ITER_RAW")
else
    ITER_TAG=$(python - <<PY 2>/dev/null
import torch
try:
    d = torch.load("$CHECKPOINT", map_location="cpu", weights_only=False)
    print(f"{int(d.get('iteration', -1)):07d}")
except Exception:
    print("unknown")
PY
)
fi
RUN_NAME=$(basename "$(dirname "$CHECKPOINT")")

# ----- 3. Build output paths and the LOG file path -----
OUTPUT_JSON="$OFFICIAL_DIR/outputs/probes/probe_adni_${RUN_NAME}_iter${ITER_TAG}.json"
FEATURES_CACHE="$OFFICIAL_DIR/outputs/probes/features_adni_${RUN_NAME}_iter${ITER_TAG}.npz"
CONFIG_FILE="$(dirname "$CHECKPOINT")/config.yaml"
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="dinov2/configs/train/fmri_vits.yaml"

LOG_OUT="$OFFICIAL_DIR/slurm_jobs/logs/probe_adni_${RUN_NAME}_iter${ITER_TAG}.out"
LOG_ERR="$OFFICIAL_DIR/slurm_jobs/logs/probe_adni_${RUN_NAME}_iter${ITER_TAG}.err"
ln -sf "$(basename "$LOG_OUT")" "$OFFICIAL_DIR/slurm_jobs/logs/probe_adni_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$OFFICIAL_DIR/slurm_jobs/logs/probe_adni_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

# ----- 4. From here, everything goes to LOG_OUT/LOG_ERR -----
echo "============================================================"
echo "  ADNI linear probe on DINOv2-fmri checkpoint"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-(local)}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  RUN_NAME:   $RUN_NAME"
echo "  Iteration:  $ITER_TAG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config:     $CONFIG_FILE"
echo "  Output:     $OUTPUT_JSON"
echo "============================================================"

# ----- 5. Install missing deps (idempotent) -----
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

# ----- 6. Run the probe -----
python scripts/probe_adni.py \
    --checkpoint "$CHECKPOINT" \
    --config-file "$CONFIG_FILE" \
    --output "$OUTPUT_JSON" \
    --features-out "$FEATURES_CACHE"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  JSON: $OUTPUT_JSON"
echo "============================================================"
