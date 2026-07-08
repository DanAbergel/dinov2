#!/bin/bash
# =====================================================================
# Re-run ALL probes: every run x every dataset x every head, to compare
# against Brain-JEPA AND NeuroSTORM in one shot.
#
#   runs     : base fourier noblock2 pool unfrozen
#   datasets : ADNI ABIDE HCP OASIS  (Brain-JEPA)  +  ADHD HCP_TASK COBRE HCP_COG (NeuroSTORM)
#   heads    : linear  +  mlp (point-3: tests 5 MLP archs x 4 alphas, keeps best by CV)
#
# Embeddings are cached per (run, dataset) -> already-probed combos are instant;
# only new datasets extract. HCP_COG reuses the HCP cache. Multiclass (HCP_TASK)
# and regression (HCP_COG) are head-agnostic -> only run once (linear).
#
# Writes probe_<run>_<ds>[_mlp].json into json_results/, then commits + pushes.
#
# Usage (GPU job — needs the dino env for the encoder):
#   sbatch -A arieljaffe tasks/v2/probe/run_all_probes.sh
#   # subsets:
#   RUNS="unfrozen" DATASETS="ADHD HCP_COG" HEADS="linear mlp" \
#       sbatch -A arieljaffe tasks/v2/probe/run_all_probes.sh
# =====================================================================

#SBATCH --job-name=probe-all
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
# (probe = forward-only, batch 1 -> ~7 GB, CPU/disk-bound; l40s is plenty.
#  Force a bigger GPU with:  sbatch --gres=gpu:h200:1 ...)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v2/probe"
RES_DIR="$TASK_DIR/json_results"
RUNS_DIR="$LAB_DIR/runs/v2"
VENV="${VENV:-$LAB_DIR/torch_env}"

RUNS="${RUNS:-base fourier noblock2 pool unfrozen}"
DATASETS="${DATASETS:-ADNI ABIDE HCP OASIS ADHD HCP_TASK COBRE HCP_COG}"
HEADS="${HEADS:-linear mlp}"
# these datasets ignore --head (own classifier) -> probe once, linear only
HEAD_AGNOSTIC="HCP_TASK HCP_COG"

export TMPDIR="$LAB_DIR/tmp"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export HOME="$LAB_DIR"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$RES_DIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

exec >"$TASK_DIR/logs/run_all_probes.out" 2>"$TASK_DIR/logs/run_all_probes.err"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "  probe-all   Node $(hostname)   $(date)"
echo "  runs:     $RUNS"
echo "  datasets: $DATASETS"
echo "  heads:    $HEADS"
echo "============================================================"

is_agnostic() { case " $HEAD_AGNOSTIC " in *" $1 "*) return 0;; *) return 1;; esac; }

for RUN in $RUNS; do
    rd="$RUNS_DIR/$RUN"
    if [ ! -f "$rd/config.yaml" ]; then
        echo ">> skip run '$RUN' (no $rd/config.yaml — not trained yet)"; continue
    fi
    for DS in $DATASETS; do
        ds=$(echo "$DS" | tr 'A-Z' 'a-z')
        for HEAD in $HEADS; do
            # head-agnostic datasets: only run the linear pass (mlp would duplicate)
            if [ "$HEAD" = "mlp" ] && is_agnostic "$DS"; then continue; fi
            suffix=""; [ "$HEAD" = "mlp" ] && suffix="_mlp"
            out="$RES_DIR/probe_${RUN}_${ds}${suffix}.json"
            echo ""
            echo "==== $RUN / $DS / $HEAD  ($(date)) ===="
            srun python -u "$TASK_DIR/probe.py" \
                --run-dir "$rd" --dataset "$DS" --head "$HEAD" --out "$out" \
                || echo "  FAILED: $RUN/$DS/$HEAD (continuing)"
        done
    done
done

echo ""
echo "==== all probes done, committing LOCALLY ($(date)) ===="
cd "$OFFICIAL_DIR"
git config user.name  "Dan Abergel"        2>/dev/null || true
git config user.email "danabergel1995@gmail.com" 2>/dev/null || true
# Shared checkout with parallel jobs: pull/push HERE breaks (races on untracked
# files + no network). Commit LOCALLY only -> linear history, no divergence.
for _ in $(seq 1 120); do [ -f .git/index.lock ] && sleep 10 || break; done
git add "$RES_DIR"/probe_*.json \
        "$TASK_DIR/logs/run_all_probes.out" "$TASK_DIR/logs/run_all_probes.err" 2>&1 || true
git commit -m "probe: full re-run (all runs x datasets x heads) — JEPA + NeuroSTORM (+logs)" 2>&1 \
    || echo "  (nothing to commit)"
echo ""
echo ">> Results committed LOCALLY. From the login node (moriah-gw), run ONCE:"
echo "     cd $OFFICIAL_DIR && git pull --no-rebase --no-edit && git push"
echo "All done: $(date)"
