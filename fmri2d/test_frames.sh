#!/bin/bash
# Ariel's two ideas together:
#   (1) MULTIPLE 2D slices per scan at multiple timepoints (--n-frames), not one temporal mean.
#   (2) LOCAL crops = GLOBAL crops: local_crops_size=224 & same scale as global -> the student's
#       "local" views are full-brain (like the teacher's globals). Fixes the useless tiny-fragment
#       dino_local term on homogeneous brains.
# Plus the winning temp schedule (0.01->0.04, warmup over half).
#
# Launch:  N_FRAMES=8 sbatch -A arieljaffe fmri2d/test_frames.sh
# Probe:   RUN_DIR=$LAB/runs/brain/brain_frames FEATURES_ROOT=$LAB/brain2d_frames \
#          PROBE_OPTS="dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048" \
#            sbatch -A arieljaffe fmri2d/probe_brain.sh
#
#SBATCH --job-name=brain-frames
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/test_frames.out
#SBATCH --error=fmri2d/test_frames.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
N_FRAMES="${N_FRAMES:-8}"
OUT_FRAMES="$LAB_DIR/brain2d_frames"
RUN="brain_frames"
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"
HCP_GLOB="$LAB_DIR/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

# 1) extract N_FRAMES timepoints per scan (re-extract if incomplete)
N_SCANS=$(ls $HCP_GLOB 2>/dev/null | wc -l)
EXPECTED=$((N_SCANS * N_FRAMES))
N_PNG=$(ls "$OUT_FRAMES/HCP" 2>/dev/null | wc -l || echo 0)
echo "scans: $N_SCANS   frames/scan: $N_FRAMES   expected PNGs: $EXPECTED   existing: $N_PNG"
if [ "$N_PNG" -lt "$EXPECTED" ]; then
    echo "=== extracting $N_FRAMES frames/scan -> $OUT_FRAMES  $(date) ==="
    python fmri2d/extract_slices.py --input "$HCP_GLOB" \
        --output "$OUT_FRAMES" --class-name HCP --axis 2 --size 224 --n-frames "$N_FRAMES"
fi
echo "frame PNGs now: $(ls "$OUT_FRAMES/HCP" 2>/dev/null | wc -l)"

# 2) train: local crops = global crops + winning temp schedule
echo "=== training (local=global) -> $OUTPUT_DIR  $(date) ==="
srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$OUT_FRAMES" \
    crops.local_crops_size=224 "crops.local_crops_scale=[0.32,1.0]" \
    teacher.warmup_teacher_temp=0.01 teacher.teacher_temp=0.04 teacher.warmup_teacher_temp_epochs=25 \
    dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048 dino.koleo_loss_weight=0
echo "=== done $(date) ==="
