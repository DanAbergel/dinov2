#!/bin/bash
# =====================================================================
# Prepare the DINOv2 ImageNet init checkpoint for fMRI transfer learning.
#
# Downloads DINOv2 ViT-S/14 reg4 weights and strips the shape-incompatible
# keys (patch_embed.*, pos_embed), saving {"model": ...} for
# student.pretrained_weights.
#
# NEEDS INTERNET -> run on the GATEWAY (not a compute node). NOT an sbatch:
#   bash tasks/prepare_dinov2_init/prepare_dinov2_init.sh
# =====================================================================

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/prepare_dinov2_init"
VENV_DIR="$LAB_DIR/torch_env"
OUT="${OUT:-$LAB_DIR/checkpoints/dinov2_vits14_reg4_fmri_init.pth}"

mkdir -p "$TASK_DIR/logs" "$(dirname "$OUT")"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$OFFICIAL_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "  Prepare DINOv2 init checkpoint   Date: $(date)"
echo "  out: $OUT"
echo "============================================================"

python3 "$TASK_DIR/prepare_dinov2_init.py" --out "$OUT" 2>&1 | tee "$TASK_DIR/logs/prepare.out"

echo ""
echo "Done: $(date)"
