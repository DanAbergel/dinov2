#!/bin/bash
# One ablation cell: TRAIN (multi-frame + local=global) THEN PROBE (sex, subject-level CV),
# in a single job. Parameterised by env vars. Submitted by run_ablations.sh.
#   env: RUN (name), PROTOS, WARMUP_TT, TT, WARMUP_EPOCHS, CV
# Log: fmri2d/logs/<RUN>.out  (the job name is set to <RUN> by the launcher -> %x).
#
#SBATCH --job-name=brain-abl
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
FRAMES="$LAB_DIR/brain2d_frames"                  # multi-frame data (pre-extracted by test_frames)
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"
RUN="${RUN:?set RUN}"
PROTOS="${PROTOS:-2048}"
WARMUP_TT="${WARMUP_TT:-0.01}"
TT="${TT:-0.04}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-25}"
CV="${CV:-5}"
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

[ -d "$FRAMES/HCP" ] || { echo "ERROR: $FRAMES not found — run test_frames.sh first to extract the multi-frame images"; exit 1; }

echo "=== ABLATION $RUN : protos=$PROTOS  temp=$WARMUP_TT->$TT (warmup $WARMUP_EPOCHS ep)  cv=$CV  $(date) ==="

# 1) TRAIN — multi-frame + local=global
srun python dinov2/train/train.py \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$FRAMES" \
    crops.local_crops_size=224 "crops.local_crops_scale=[0.32,1.0]" \
    teacher.warmup_teacher_temp="$WARMUP_TT" teacher.teacher_temp="$TT" teacher.warmup_teacher_temp_epochs="$WARMUP_EPOCHS" \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS" dino.koleo_loss_weight=0

# 2) PROBE — sex, subject-level k-fold CV (same job, right after)
LABELS="$TMPDIR/HCP_YA_subjects.csv"
git show origin/fmri-multi-source:data/HCP_YA_subjects.csv > "$LABELS"
echo "=== PROBE $RUN  $(date) ==="
srun python fmri2d/probe_brain.py \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --features-root "$FRAMES" --labels-csv "$LABELS" --label-col Gender --test-frac 0.2 --cv "$CV" \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS"
echo "=== $RUN DONE  $(date) ==="
