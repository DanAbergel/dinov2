#!/bin/bash
# =====================================================================
# Probe ABLATIONS — ALL on the 'base' run (best pretraining).
#
#   A) Aggregation ablation : how per-window CLS are pooled to one vector
#        - mean      (384-d)   [current default]
#        - mean_std  (768-d)   [mean ‖ std, adds temporal dynamics]
#      head = linear, over the main axes.
#
#   B) MLP-architecture ablation : one probe per MLP arch (frozen encoder),
#      agg = mean, so we SEE each architecture's test score (not just the
#      CV-selected winner). Archs: 128 / 256 / 256,128 / 512,256 / 512,256,128.
#
# Output json names encode the ablation so nothing collides:
#   probe_base_<ds>_agg-<mean|mean_std>.json
#   probe_base_<ds>_mlp-<arch>.json
# =====================================================================

#SBATCH --job-name=probe-abl
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v2/probe"
RES_DIR="$TASK_DIR/json_results"
RUN_DIR="$LAB_DIR/runs/v2/base"                 # ALL ablations on base
VENV="${VENV:-$LAB_DIR/torch_env}"

# main axes for the ablations (data all present); override with DATASETS=...
DATASETS="${DATASETS:-ADNI ABIDE HCP}"
AGGS="${AGGS:-mean mean_std}"
MLP_ARCHS="${MLP_ARCHS:-128 256 256,128 512,256 512,256,128}"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"
export HOME="$LAB_DIR"; export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"; export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$RES_DIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exec >"$TASK_DIR/logs/run_probe_ablations.out" 2>"$TASK_DIR/logs/run_probe_ablations.err"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

[ -f "$RUN_DIR/config.yaml" ] || { echo "ERROR: base not trained ($RUN_DIR)"; exit 1; }
echo "=== probe ablations on BASE   $(hostname)   $(date) ==="
echo "datasets=$DATASETS  aggs=$AGGS  mlp_archs=$MLP_ARCHS"

probe() {  # $1=dataset $2=extra-args $3=out-tag
    local DS="$1" EXTRA="$2" TAG="$3" ds
    ds=$(echo "$DS" | tr 'A-Z' 'a-z')
    echo ""; echo "==== $DS  [$TAG]  ($(date)) ===="
    srun python -u "$TASK_DIR/probe.py" --run-dir "$RUN_DIR" --dataset "$DS" \
        --out "$RES_DIR/probe_base_${ds}_${TAG}.json" $EXTRA \
        || echo "  FAILED: $DS [$TAG] (continuing)"
}

# --- A) aggregation ablation (linear head) ---
for DS in $DATASETS; do
    for AGG in $AGGS; do
        probe "$DS" "--head linear --agg $AGG" "agg-${AGG}"
    done
done

# --- B) MLP-architecture ablation (agg=mean, one arch each) ---
for DS in $DATASETS; do
    for ARCH in $MLP_ARCHS; do
        probe "$DS" "--head mlp --agg mean --mlp-arch $ARCH" "mlp-${ARCH//,/x}"
    done
done

echo ""; echo "==== committing ($(date)) ===="
cd "$OFFICIAL_DIR"
git config user.name  "Dan Abergel"              2>/dev/null || true
git config user.email "danabergel1995@gmail.com" 2>/dev/null || true
for _ in $(seq 1 60); do [ -f .git/index.lock ] && sleep 10 || break; done
git add "$RES_DIR"/probe_base_*_agg-*.json "$RES_DIR"/probe_base_*_mlp-*.json 2>&1 || true
git commit -m "probe ablations on base: aggregation (mean/mean_std) + MLP archs" 2>&1 \
    || echo "  (nothing to commit)"
git stash -u 2>&1 || true
git pull --no-rebase --no-edit 2>&1 || true
git stash pop 2>&1 || true
git push 2>&1 || echo "  PUSH FAILED -> run 'git push' from the login node"
echo "All done: $(date)"
