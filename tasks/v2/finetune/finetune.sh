#!/bin/bash
# =====================================================================
# SLURM job — fine-tune (SFT) a pretrained encoder on a downstream task
# (point 4, Brain-JEPA's headline protocol). Unfreezes the encoder and
# trains end-to-end + a linear head, leakage-free 70:30, mean±std over seeds.
#
# Usage:
#   # from the DINOv2 init (transformer=ImageNet, patchify+pos RANDOM; no-SSL baseline) — runs NOW:
#   FROM_INIT=1 DATASET=ADNI DEPTH=all SEEDS=0,1,2 sbatch -A arieljaffe tasks/v2/finetune/finetune.sh
#   # from our SSL-pretrained run (needs the run trained):
#   RUN=base DATASET=ADNI DEPTH=all SEEDS=0,1,2 sbatch -A arieljaffe tasks/v2/finetune/finetune.sh
#   RUNS_DIR=$LAB/runs/v1 RUN=base DATASET=HCP DEPTH=last3 sbatch -A arieljaffe tasks/v2/finetune/finetune.sh
#   SMOKE=1 FROM_INIT=1 DATASET=ADNI sbatch -A arieljaffe tasks/v2/finetune/finetune.sh   # pipeline check
# Result JSON -> tasks/v2/finetune/json_results/ , auto-committed+pushed.
# =====================================================================

#SBATCH --job-name=ft
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/v2/finetune"
VENV_DIR="$LAB_DIR/torch_env"

RUN="${RUN:-base}"
DATASET="${DATASET:-ADNI}"
DEPTH="${DEPTH:-last3}"                       # head | last3 | all
SEEDS="${SEEDS:-0}"
CKPT="${CKPT:-model_final.rank_0.pth}"
FROM_INIT="${FROM_INIT:-0}"                   # 1 = fine-tune from DINOv2 init (transformer
                                              #     ImageNet, patchify+pos RANDOM; no-SSL baseline)
RES_DIR="$OFFICIAL_DIR/tasks/v2/finetune/json_results"
mkdir -p "$TASK_DIR/logs" "$RES_DIR"

if [ "$FROM_INIT" = "1" ]; then
    NAME="dinov2init"
    SRC_ARGS="--from-init"
else
    RUNS_DIR="${RUNS_DIR:-$LAB_DIR/runs/v2}"   # base/fourier(v1) via RUNS_DIR=$LAB/runs/v1
    NAME="$RUN"
    SRC_ARGS="--run-dir $RUNS_DIR/$RUN --checkpoint $CKPT"
fi
ds=$(echo "$DATASET" | tr 'A-Z' 'a-z')
SMOKE_TAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_TAG="_smoke"
LOG="$TASK_DIR/logs/ft_${NAME}_${DATASET}_${DEPTH}${SMOKE_TAG}.out"
OUT="$RES_DIR/finetune_${NAME}_${ds}_${DEPTH}.json"
exec >"$LOG" 2>&1

export TMPDIR="$LAB_DIR/tmp"
export HOME="$LAB_DIR"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Fine-tune  src=$NAME dataset=$DATASET depth=$DEPTH seeds=$SEEDS from_init=$FROM_INIT  Node=$(hostname)  $(date)"
srun python -u tasks/v2/finetune/finetune.py \
    $SRC_ARGS --dataset "$DATASET" --depth "$DEPTH" \
    --seeds "$SEEDS" --out "$OUT" ${SMOKE:+--smoke}
echo "Fine-tune done: $(date)"

# commit + push the result JSON (skip for smoke). Parallel-safe (wait out index.lock).
if [ "${SMOKE:-0}" != "1" ] && [ -f "$OUT" ]; then
    cd "$OFFICIAL_DIR"
    for _ in $(seq 1 60); do [ -f .git/index.lock ] && sleep 10 || break; done
    git add "$OUT" 2>&1 || true
    git commit -m "auto: finetune ${NAME} ${DATASET} ${DEPTH}" 2>&1 || echo "  (nothing to commit)"
    git pull --no-rebase --no-edit 2>&1 || true
    git push 2>&1 && echo "  pushed OK" || echo "  PUSH FAILED -> git push from the login node"
fi
echo "All done: $(date)"
