#!/bin/bash
# OVERFIT SANITY (fMRI): train on ONE brain image and see how low the loss can go.
# If even a single image can't drive the loss down, the problem is mechanical
# (centering / temperature), not the data. Config is tuned to memorise:
#   1 image (IMAGENETTE_OVERFIT_N=1), koleo=0 (else log(0)->NaN on identical batch),
#   few prototypes (low floor), sharp constant teacher_temp, higher LR, no last-layer
#   freeze, many iters. Augmentations stay ON (standard DINO views of the one image).
# No probe (1 image -> probe meaningless). Just watch total_loss.
#
# Launch:  sbatch -A arieljaffe fmri2d/overfit_one.sh
# Watch:   tail -f fmri2d/logs/overfit1.out | grep total_loss
#
#SBATCH --job-name=overfit1
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/logs/%x.out
#SBATCH --error=fmri2d/logs/%x.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"
RUN="${RUN:-overfit1}"
DATA_ROOT="${DATA_ROOT:-$LAB_DIR/brain2d}"   # source folder; only N images are kept
N="${N:-1}"                                  # number of images to overfit
BATCH="${BATCH:-32}"                          # >1 so centering has a batch of augmented views
PROTOS="${PROTOS:-128}"                       # few prototypes -> low loss floor
TT="${TT:-0.02}"                              # final teacher temperature
WARMUP_TT="${WARMUP_TT:-$TT}"                 # start teacher temperature (defaults to TT = constant)
WTE="${WTE:-1}"                               # teacher-temp warmup epochs
LR="${LR:-0.01}"
EPOCHS="${EPOCHS:-100}"; OEL="${OEL:-100}"    # 100*100 = 10000 iters
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

export IMAGENETTE_OVERFIT_N="$N"             # ImageFolder reader keeps only N images (prints them)
[ -d "$DATA_ROOT/HCP" ] || { echo "ERROR: $DATA_ROOT not found"; exit 1; }

echo "=== OVERFIT $RUN : data=$(basename "$DATA_ROOT") N=$N batch=$BATCH protos=$PROTOS temp=$WARMUP_TT->$TT (warmup $WTE ep) lr=$LR iters=$((EPOCHS*OEL))  $(date) ==="

srun python dinov2/train/train.py \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$DATA_ROOT" \
    train.batch_size_per_gpu="$BATCH" \
    dino.koleo_loss_weight=0 \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS" \
    teacher.teacher_temp="$TT" teacher.warmup_teacher_temp="$WARMUP_TT" teacher.warmup_teacher_temp_epochs="$WTE" \
    optim.base_lr="$LR" optim.warmup_epochs=1 optim.freeze_last_layer_epochs=0 \
    optim.epochs="$EPOCHS" train.OFFICIAL_EPOCH_LENGTH="$OEL"

echo "=== $RUN DONE  $(date) ==="
