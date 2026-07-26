#!/bin/bash
# Neighbor-slice SSL test. Global crops = 2 augmented views of slice z (standard DINO).
# Local crops = 8 neighbor slices z + k*stride (k in -4..3), NO augmentation.
# A linear sex probe runs every PROBE_EVERY iters (and at iter 0); a PROBE EVOLUTION
# summary is printed at the end of the SAME .out file.
#
#   STRIDE=1 RUN=nb_s1 sbatch -A arieljaffe -J nb_s1 fmri2d/neighbor_test.sh   # truly adjacent
#   STRIDE=2 RUN=nb_s2 sbatch -A arieljaffe -J nb_s2 fmri2d/neighbor_test.sh   # a bit more spaced
#   watch:  tail -f fmri2d/logs/nb_s1.out
#
#SBATCH --job-name=nb-slice
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
RUN="${RUN:?set RUN}"
STRIDE="${STRIDE:-1}"                               # spacing between neighbor slices (1 = adjacent)
DENSE_POOL="${DENSE_POOL:-$LAB_DIR/brain2d_dense}"  # consecutive-slice pool (extract_dense_pool.sh)
PROBE_DATA="${PROBE_DATA:-$LAB_DIR/brain2d}"        # fixed-slice, 1/subject -> comparable probe
PROTOS="${PROTOS:-2048}"
WARMUP_TT="${WARMUP_TT:-0.01}"; TT="${TT:-0.04}"; WARMUP_EPOCHS="${WARMUP_EPOCHS:-25}"
LOCAL_SIZE="${LOCAL_SIZE:-224}"                     # neighbor slices resized to this (full view)
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"

export DINO_NEIGHBOR_STRIDE="$STRIDE"
export DINO_NEIGHBOR_MODE="${DINO_NEIGHBOR_MODE:-z}"   # "z" (subject/z) or "slice_time" (subject+slice / time)
export PROBE_ROOT="$PROBE_DATA"; export PROBE_EVERY="${PROBE_EVERY:-2000}"
export PROBE_LABEL_COL=Gender; export PROBE_CV=5; export PROBE_AVGPOOL=1
export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" fmri2d/logs
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"   # repo root -> import fmri2d.online_probe

# labels for the periodic probe (sex), pulled from the fMRI branch (no network, no commit)
LABELS="$TMPDIR/HCP_YA_subjects.csv"
git show origin/fmri-multi-source:data/HCP_YA_subjects.csv > "$LABELS"
export PROBE_LABELS="$LABELS"

[ -d "$DENSE_POOL/HCP" ] || { echo "ERROR: dense pool $DENSE_POOL not found — run extract_dense_pool.sh first"; exit 1; }
[ -d "$PROBE_DATA/HCP" ] || { echo "ERROR: probe data $PROBE_DATA/HCP not found"; exit 1; }

echo "=== NEIGHBOR $RUN : stride=$STRIDE pool=$(basename "$DENSE_POOL") local_size=$LOCAL_SIZE protos=$PROTOS probe_every=$PROBE_EVERY  $(date) ==="

srun python dinov2/train/train.py --no-resume \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    train.dataset_path="SliceNeighborsFolder:root=$DENSE_POOL" \
    crops.local_crops_size="$LOCAL_SIZE" crops.local_crops_number=8 \
    teacher.warmup_teacher_temp="$WARMUP_TT" teacher.teacher_temp="$TT" teacher.warmup_teacher_temp_epochs="$WARMUP_EPOCHS" \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS" dino.koleo_loss_weight=0

echo "=== $RUN DONE  $(date) ==="
