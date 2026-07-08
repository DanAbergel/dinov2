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
#SBATCH --gres=gpu:l40s:1
# (probe = forward-only, batch 1 -> ~7 GB, CPU/disk-bound; l40s is plenty and
#  schedules faster. Force a bigger GPU with:  sbatch --gres=gpu:h200:1 ...)
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

# ALL datasets matching the two SOTAs' predicted labels:
#   Brain-JEPA : ADNI ABIDE HCP OASIS
#   NeuroSTORM : ADHD COBRE HCP_TASK HCP_COG (+ ABIDE/HCP shared)
# (missing data -> that dataset is skipped gracefully). Override with DATASETS=...
DATASETS="${DATASETS:-ADNI ABIDE HCP OASIS ADHD COBRE HCP_TASK HCP_COG}"
AGGS="${AGGS:-mean mean_std}"
MLP_ARCHS="${MLP_ARCHS:-128 256 256,128 512,256 512,256,128}"
# these use their own classifier (multiclass / regression) -> --head/--mlp-arch
# are ignored, so skip them in the MLP-arch ablation (they'd just duplicate).
HEAD_AGNOSTIC="HCP_TASK HCP_COG"
is_agnostic() { case " $HEAD_AGNOSTIC " in *" $1 "*) return 0;; *) return 1;; esac; }

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
# skip multiclass/regression datasets (head-agnostic -> would just duplicate).
for DS in $DATASETS; do
    if is_agnostic "$DS"; then
        echo ""; echo "-- skip MLP-arch ablation for $DS (head-agnostic)"; continue
    fi
    for ARCH in $MLP_ARCHS; do
        probe "$DS" "--head mlp --agg mean --mlp-arch $ARCH" "mlp-${ARCH//,/x}"
    done
done

echo ""; echo "==== committing results LOCALLY ($(date)) ===="
cd "$OFFICIAL_DIR"
git config user.name  "Dan Abergel"              2>/dev/null || true
git config user.email "danabergel1995@gmail.com" 2>/dev/null || true
# Several jobs share ONE checkout. Doing pull/push HERE is what breaks:
#   - pull races with other jobs' untracked result files (merge aborts)
#   - push needs network the GPU node lacks + causes remote divergence
# So: commit LOCALLY only (linear history, no divergence). Wait out any index.lock
# held by a parallel job, stage ONLY our own files, commit. Push ONCE from login node.
for _ in $(seq 1 120); do [ -f .git/index.lock ] && sleep 10 || break; done
git add "$RES_DIR"/probe_base_*_agg-*.json "$RES_DIR"/probe_base_*_mlp-*.json \
        "$TASK_DIR/logs/run_probe_ablations.out" "$TASK_DIR/logs/run_probe_ablations.err" 2>&1 || true
git commit -m "probe ablations on base: aggregation (mean/mean_std) + MLP archs (+logs)" 2>&1 \
    || echo "  (nothing to commit)"
echo ""
echo ">> Results committed LOCALLY. From the login node (moriah-gw), run ONCE:"
echo "     cd $OFFICIAL_DIR && git pull --no-rebase --no-edit && git push"
echo "All done: $(date)"
