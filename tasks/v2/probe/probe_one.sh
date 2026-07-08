#!/bin/bash
# =====================================================================
# ONE probe = ONE SLURM job (for parallel fan-out). Parametrized by env vars:
#   RUN       (default base)      DATASET (required)
#   HEAD      (default linear)    AGG     (default mean)   MLP_ARCH (optional)
#   OUTTAG    (optional filename tag; else derived from head/agg/arch)
#
# Writes ONE probe_<run>_<ds>_<tag>.json. Does NO git (many parallel jobs share
# one checkout -> commit ONCE from the login node afterwards).
#
# Launch directly, or via launch_mlp_ablation.sh (fans out one job per combo):
#   sbatch -A arieljaffe --export=ALL,DATASET=ADNI,HEAD=mlp,MLP_ARCH=256,128 \
#       tasks/v2/probe/probe_one.sh
# =====================================================================

#SBATCH --job-name=probe1
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v2/probe"
RES_DIR="$TASK_DIR/json_results"
VENV="${VENV:-$LAB_DIR/torch_env}"

RUN="${RUN:-base}"
DATASET="${DATASET:?set DATASET=ADNI|ABIDE|HCP|ADHD|COBRE|...}"
HEAD="${HEAD:-linear}"
AGG="${AGG:-mean}"
MLP_ARCH="${MLP_ARCH:-}"
ds=$(echo "$DATASET" | tr 'A-Z' 'a-z')

# output tag: mlp-<arch> > agg-<agg> > <head>
if [ -n "${OUTTAG:-}" ]; then TAG="$OUTTAG"
elif [ -n "$MLP_ARCH" ];   then TAG="mlp-${MLP_ARCH//,/x}"
elif [ "$AGG" != "mean" ]; then TAG="agg-${AGG}"
else TAG="$HEAD"; fi
OUT="$RES_DIR/probe_${RUN}_${ds}_${TAG}.json"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"
export HOME="$LAB_DIR"; export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"; export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$RES_DIR" "$TASK_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exec >"$TASK_DIR/logs/probe1_${RUN}_${ds}_${TAG}.out" \
     2>"$TASK_DIR/logs/probe1_${RUN}_${ds}_${TAG}.err"

source "$VENV/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

echo "== probe_one  RUN=$RUN DS=$DATASET HEAD=$HEAD AGG=$AGG ARCH=${MLP_ARCH:-none}  $(date) =="
EXTRA=""; [ -n "$MLP_ARCH" ] && EXTRA="--mlp-arch $MLP_ARCH"
srun python -u "$TASK_DIR/probe.py" \
    --run-dir "$LAB_DIR/runs/v2/$RUN" --dataset "$DATASET" \
    --head "$HEAD" --agg "$AGG" $EXTRA --out "$OUT"
echo "done -> $OUT  ($(date))"
