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
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"
RUN="${RUN:?set RUN}"
PROTOS="${PROTOS:-2048}"
WARMUP_TT="${WARMUP_TT:-0.01}"
TT="${TT:-0.04}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-25}"
CV="${CV:-5}"
DATA_ROOT="${DATA_ROOT:-$LAB_DIR/brain2d_frames}"  # ImageFolder root (multi-frame by default; brain2d = 1 mean image/scan)
LOCAL_GLOBAL="${LOCAL_GLOBAL:-1}"                   # 1 = local crops = global (full-view) ; 0 = default 96px local
LOCAL_EQ_GLOBAL="${LOCAL_EQ_GLOBAL:-0}"            # 1 = DIAGNOSTIC: local crops = pixel-identical copy of global crop
DINO_WEIGHT="${DINO_WEIGHT:-1}"                    # 0 = iBOT-dominant (drop the DINO/CLS discriminative term)
CENTERING="${CENTERING:-centering}"               # "centering" (default) or "sinkhorn_knopp" (forces spread) 1
LR="${LR:-}"                                       # override optim.base_lr (e.g. 0.001); empty = config default (0.004)
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"

export DINO_LOCAL_EQ_GLOBAL="$LOCAL_EQ_GLOBAL"    # read in dinov2/data/augmentations.py (propagated to srun)
export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

[ -d "$DATA_ROOT/HCP" ] || { echo "ERROR: $DATA_ROOT not found — extract it first (test_frames.sh for multi-frame, extract_hcp.sh for mean)"; exit 1; }

# local=global crops only when LOCAL_GLOBAL=1 (array keeps the [..] scale from bash globbing/splitting)
CROPS=()
[ "$LOCAL_GLOBAL" = "1" ] && CROPS=(crops.local_crops_size=224 'crops.local_crops_scale=[0.32,1.0]')
LR_OPT=()
[ -n "$LR" ] && LR_OPT=(optim.base_lr="$LR")

echo "=== ABLATION $RUN : data=$(basename "$DATA_ROOT") local_global=$LOCAL_GLOBAL local_eq_global=$LOCAL_EQ_GLOBAL protos=$PROTOS temp=$WARMUP_TT->$TT (warmup $WARMUP_EPOCHS) lr=${LR:-default} cv=$CV  $(date) ==="

# 1) TRAIN
srun python dinov2/train/train.py --no-resume \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$DATA_ROOT" \
    "${CROPS[@]}" "${LR_OPT[@]}" \
    teacher.warmup_teacher_temp="$WARMUP_TT" teacher.teacher_temp="$TT" teacher.warmup_teacher_temp_epochs="$WARMUP_EPOCHS" \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS" dino.koleo_loss_weight=0 dino.loss_weight="$DINO_WEIGHT" \
    train.centering="$CENTERING"

# 2) PROBE — sex, subject-level k-fold CV (same job, right after)
LABELS="$TMPDIR/HCP_YA_subjects.csv"
git show origin/fmri-multi-source:data/HCP_YA_subjects.csv > "$LABELS"
echo "=== PROBE $RUN  $(date) ==="
AVGPOOL="${AVGPOOL:-1}"                            # 1 = official DINOv2 repr (CLS ++ avgpool patch tokens); "" = CLS-only
srun python fmri2d/probe_brain.py \
    --config-file "$CONFIG" --output-dir "$OUTPUT_DIR" \
    --features-root "$DATA_ROOT" --labels-csv "$LABELS" --label-col Gender --test-frac 0.2 --cv "$CV" \
    ${AVGPOOL:+--avgpool} \
    dino.head_n_prototypes="$PROTOS" ibot.head_n_prototypes="$PROTOS"
echo "=== $RUN DONE  $(date) ==="
