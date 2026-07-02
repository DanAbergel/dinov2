#!/bin/bash
# =====================================================================
# SLURM job — train the V2 fMRI foundation model (official dinov2 fork).
#
# 5-source corpus (HCP/ABIDE/OASIS/AOMIC/ADNI, 4627 scans) with:
#   - online TR harmonization to 0.72s + T_fixed=270 window (MixedFMRIDataset)
#   - ProportionalBatchSampler (HCP4/ABIDE4/OASIS4/ADNI3/AOMIC1 = 16)
#   - masking-only augmentation (full-image crops + per-token random masking)
#   - learned spatial pos (Fourier is a LATER ablation)
#
# Config: dinov2/configs/train/fmri_vits.yaml
#
# IMPORTANT — verify these for your cluster before launching:
#   VENV  : the dinov2 env (xformers etc.), NOT the data-tasks torch_env.
#           Defaults to the in-repo dinov2/env_dino; override with VENV=...
#   GRES  : GPU type/count. Defaults to gpu:l40s:1; override with GRES=...
#
# Each run TRAINS, then auto-runs the leakage-free linear probe on ABIDE/ADNI/HCP
# (writes probe_<ds>.json in the run dir). Set PROBE=0 to skip the probe step.
#
# Smoke test FIRST (cheap, ~10 iters, catches build/shape bugs; no probe):
#   SMOKE=1 sbatch -A arieljaffe tasks/v2/train/train.sh
# Pretraining ablation runs (one factor each):
#   RUN=base                 sbatch -A arieljaffe tasks/v2/train/train.sh   # reference
#   RUN=fourier  FOURIER=1   sbatch -A arieljaffe tasks/v2/train/train.sh   # Fourier spatial pos
#   RUN=noblock2 NOBLOCK2=1  sbatch -A arieljaffe tasks/v2/train/train.sh   # point 2: drop block_2
#   RUN=pool     NOBLOCK2=1 POOL=1 sbatch -A arieljaffe tasks/v2/train/train.sh  # point 2: + temporal AvgPool
# =====================================================================

#SBATCH --job-name=fmri-v2
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v2/train"
CONFIG="$OFFICIAL_DIR/dinov2/configs/train/fmri_vits.yaml"

# --- cluster-specific (override on the command line) ---------------------
VENV="${VENV:-$LAB_DIR/torch_env}"                 # dinov2 training env
RUN_NAME="${RUN_NAME:-${RUN:-base}}"               # accept RUN or RUN_NAME
OUTPUT_DIR="${OUTPUT_DIR:-$LAB_DIR/runs/v2/$RUN_NAME}"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# $HOME (/sci/home) is not writable on compute nodes; Triton/xformers JIT-compile
# CUDA kernels and cache them under ~/.triton -> PermissionError. Point HOME and
# the JIT caches at the writable lab dir.
export HOME="$LAB_DIR"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

LOG_OUT="$TASK_DIR/logs/${RUN_NAME}.out"
LOG_ERR="$TASK_DIR/logs/${RUN_NAME}.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

# SMOKE=1 -> tiny run to validate the pipeline (build, 1 epoch x 10 iters).
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=10 evaluation.eval_period_iterations=0"
fi
# Optional batch override (config default 2x8=16 is safe on any GPU incl. L40s).
# On h200 use BATCH_PER_GPU=16 GRAD_ACCUM=1 (effective batch unchanged at 16, but
# one forward instead of 8 -> faster; ~55-65 GB of h200's 141 GB). Note: the
# proportional sampler caps batch_size_per_gpu at sum(quota)=16 (must divide 16).
[ -n "${BATCH_PER_GPU:-}" ] && EXTRA="$EXTRA train.batch_size_per_gpu=${BATCH_PER_GPU}"
[ -n "${GRAD_ACCUM:-}" ]    && EXTRA="$EXTRA optim.grad_accum_steps=${GRAD_ACCUM}"
# FOURIER=1 -> Fourier spatial positional encoding instead of the learned table.
# NOBLOCK2=1 / POOL=1 -> the point-2 architecture ablations (drop block_2 /
# temporal AvgPool instead of strided conv). Pair each with a distinct RUN_NAME.
[ "${FOURIER:-0}" = "1" ]   && EXTRA="$EXTRA student.fmri_fourier_pos=true"
[ "${NOBLOCK2:-0}" = "1" ]  && EXTRA="$EXTRA student.fmri_remove_block2=true"
[ "${POOL:-0}" = "1" ]      && EXTRA="$EXTRA student.fmri_temporal_pool=true"
# OVERRIDES -> any extra dinov2 config overrides, space-separated, e.g.
#   OVERRIDES="optim.base_lr=1e-3 optim.freeze_pretrained=fmri_only"
[ -n "${OVERRIDES:-}" ]     && EXTRA="$EXTRA ${OVERRIDES}"

echo "============================================================"
echo "  fMRI V2 training   Node: $(hostname)   Date: $(date)"
echo "  venv:    $VENV"
echo "  config:  $CONFIG"
echo "  output:  $OUTPUT_DIR"
echo "  smoke:   ${SMOKE:-0}   extra: $EXTRA"
echo "============================================================"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Launch with srun (NOT torchrun): dinov2's distributed.enable() takes the SLURM
# path when in a SLURM job, and srun + --ntasks sets SLURM_NTASKS/PROCID/LOCALID.
# For multi-GPU later: bump --gres + --ntasks (one task per GPU); the
# ProportionalInfiniteSampler reads rank/size from this same SLURM env.
srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA

echo ""
echo "Training done: $(date)"

# ---------------------------------------------------------------------------
# Auto linear-probe once training finishes (skipped for SMOKE runs). Reuses the
# same GPU/job. Extracts embeddings from the just-trained encoder and fits the
# leakage-free linear probe (70:30, CV-on-train) on each downstream dataset,
# writing probe_<ds>.json into the run dir. Set PROBE=0 to skip.
if [ "${SMOKE:-0}" != "1" ] && [ "${PROBE:-1}" = "1" ]; then
    RES_DIR="$OFFICIAL_DIR/tasks/v2/probe/json_results"
    mkdir -p "$RES_DIR"
    for D in ABIDE ADNI HCP; do
        echo ""
        echo "==== auto linear-probe: $D  ($(date)) ===="
        ds=$(echo "$D" | tr 'A-Z' 'a-z')
        # write the result JSON DIRECTLY into the versioned task folder (--out)
        srun python -u "$OFFICIAL_DIR/tasks/v2/probe/probe.py" \
            --run-dir "$OUTPUT_DIR" --dataset "$D" --head linear \
            --out "$RES_DIR/probe_${RUN_NAME}_${ds}.json" || \
            echo "  probe $D FAILED (continuing)"
    done
    echo ""
    echo "Probes done: $(date)"

    # commit + push the results JSON so they land in the repo automatically.
    # Best-effort: GPU compute nodes often have no network -> push may fail, and
    # the commit stays local for you to push from the login node.
    echo ""
    echo "==== committing probe results ($(date)) ===="
    cd "$OFFICIAL_DIR"
    git add "$RES_DIR" 2>&1 || true
    git commit -m "auto: probe results for ${RUN_NAME}" 2>&1 || echo "  (nothing to commit)"
    git pull --no-rebase --no-edit 2>&1 || true
    if git push 2>&1; then
        echo "  results pushed OK"
    else
        echo "  PUSH FAILED (compute node likely has no network) -> run 'git push' from the login node"
    fi
fi

echo "All done: $(date)"
