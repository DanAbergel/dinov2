#!/bin/bash
# TEST 1: per-voxel temporal z-score normalisation (tSNR = mean/std) of the HCP scans,
# then train 2D DINOv2 on those images (winning config: teacher_temp=0.02).
# Extraction is cached in brain2d_tsnr/ (skipped on re-runs). Probe afterwards with:
#   RUN_DIR=$LAB/runs/brain/brain_ztime FEATURES_ROOT=$LAB/brain2d_tsnr \
#   PROBE_OPTS="dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048" \
#     sbatch -A arieljaffe fmri2d/probe_brain.sh
#
#SBATCH --job-name=brain-test1-ztime
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/test1_ztime.out
#SBATCH --error=fmri2d/test1_ztime.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OUT_TSNR="$LAB_DIR/brain2d_tsnr"
RUN="brain_ztime"
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

# 1) extract with tSNR normalisation (per-voxel temporal mean/std), cached
if [ ! -d "$OUT_TSNR/HCP" ]; then
    echo "=== extracting HCP with --tnorm tsnr -> $OUT_TSNR  $(date) ==="
    python fmri2d/extract_slices.py --input "$HCP_GLOB" \
        --output "$OUT_TSNR" --class-name HCP --axis 2 --size 224 --tnorm tsnr
fi
echo "tSNR PNGs: $(ls "$OUT_TSNR/HCP" 2>/dev/null | wc -l)"

# 2) train 2D DINOv2 on the tSNR images (winning small-scale config)
echo "=== training -> $OUTPUT_DIR  $(date) ==="
srun python dinov2/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    train.dataset_path="ImageFolder:root=$OUT_TSNR" \
    teacher.teacher_temp=0.02 teacher.warmup_teacher_temp=0.04 \
    dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048 dino.koleo_loss_weight=0
echo "=== done $(date) ==="
