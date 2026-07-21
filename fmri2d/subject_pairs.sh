#!/bin/bash
# Train DINO where the positive pair = 2 DIFFERENT slices of the SAME subject
# (SubjectSliceFolder + MultiSliceAugmentationDINO), then probe (sex, subject-level CV).
# Goal: force invariance to slice position so the model learns SUBJECT features, not slice height.
# Needs a pool of many slices per subject -> extract first with:
#     N=24 OUT=$LAB/brain2d_pool sbatch -A arieljaffe fmri2d/extract_allsubj.sh
#
# Launch:  sbatch -A arieljaffe fmri2d/subject_pairs.sh
# Watch:   tail -f fmri2d/logs/subjpairs.out | grep -E 'teacher_entropy_ratio|CV accuracy'
#
#SBATCH --job-name=subjpairs
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
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"
RUN="${RUN:-subjpairs}"
PROTOS="${PROTOS:-4096}"
WARMUP_TT="${WARMUP_TT:-0.04}"; TT="${TT:-0.02}"; WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
CV="${CV:-5}"
DATA_ROOT="${DATA_ROOT:-$LAB_DIR/brain2d_pool}"     # pool of many slices per subject
LOCAL_GLOBAL="${LOCAL_GLOBAL:-1}"
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

[ -d "$DATA_ROOT" ] || { echo "ERROR: $DATA_ROOT not found — extract a slice pool first (extract_allsubj.sh N=24 OUT=$DATA_ROOT)"; exit 1; }

CROPS=()
[ "$LOCAL_GLOBAL" = "1" ] && CROPS=(crops.local_crops_size=224 'crops.local_crops_scale=[0.32,1.0]')

echo "=== SUBJECT-PAIRS $RUN : data=$(basename "$DATA_ROOT") local_global=$LOCAL_GLOBAL protos=$PROTOS temp=$WARMUP_TT->$TT (warmup $WARMUP_EPOCHS) cv=$CV  $(date) ==="

# 1) TRAIN — positive pair = 2 slices of the same subject
srun python dinov2/train/train.py \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    train.dataset_path="SubjectSliceFolder:root=$DATA_ROOT" \
    "${CROPS[@]}" \
    teacher.warmup_teacher_temp="$WARMUP_TT" teacher.teacher_temp="$TT" teacher.warmup_teacher_temp_epochs="$WARMUP_EPOCHS" \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS" dino.koleo_loss_weight=0

# 2) PROBE — sex, subject-level k-fold CV (reads the same PNGs individually via ImageFolder)
LABELS="$TMPDIR/HCP_YA_subjects.csv"
git show origin/fmri-multi-source:data/HCP_YA_subjects.csv > "$LABELS"
echo "=== PROBE $RUN  $(date) ==="
srun python fmri2d/probe_brain.py \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --features-root "$DATA_ROOT" --labels-csv "$LABELS" --label-col Gender --test-frac 0.2 --cv "$CV" \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS"
echo "=== $RUN DONE  $(date) ==="
