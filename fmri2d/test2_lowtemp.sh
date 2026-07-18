#!/bin/bash
# TEST 2: sharpen the teacher even more (teacher_temp=0.01, vs the 0.02 winner) on the
# existing (temporal-mean) HCP images. Does a sharper target give a better probe?
# ⚠️ Very low teacher_temp risks COLLAPSE -> always confirm with the probe (a loss that
#    plunges + a probe near 50% = collapse, not learning).
# Probe afterwards with:
#   RUN_DIR=$LAB/runs/brain/brain_tt001 FEATURES_ROOT=$LAB/brain2d \
#   PROBE_OPTS="dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048" \
#     sbatch -A arieljaffe fmri2d/probe_brain.sh
#
#SBATCH --job-name=brain-test2-tt001
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/test2_lowtemp.out
#SBATCH --error=fmri2d/test2_lowtemp.out
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
BRAIN_ROOT="$LAB_DIR/brain2d"                    # existing temporal-mean images
RUN="brain_tt001"
OUTPUT_DIR="$LAB_DIR/runs/brain/$RUN"
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

echo "=== training (teacher_temp=0.01) -> $OUTPUT_DIR  $(date) ==="
srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$BRAIN_ROOT" \
    teacher.teacher_temp=0.01 teacher.warmup_teacher_temp=0.04 \
    dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048 dino.koleo_loss_weight=0
echo "=== done $(date) ==="
