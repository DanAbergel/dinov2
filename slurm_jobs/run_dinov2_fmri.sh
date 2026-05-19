#!/bin/bash
# =====================================================================
# SLURM Job — DINOv2 OFFICIAL SSL on fMRI (HCP)
#
# Runs the upstream `dinov2.train.train` entry point with our fMRI
# config. The official algorithm is reused as-is (SSLMetaArch,
# do_train, build_schedulers, get_params_groups_with_decay, FSDP,
# fp16 + ShardedGradScaler, PeriodicCheckpointer). The only fMRI
# touch-points are:
#   - dinov2/data/fmri_data.py          (new: HCP/ADNI datasets, MultiCrop3D)
#   - dinov2/layers/patch_embed_3d_plus_1d.py (new: 3D Conv + 1D Conv)
#   - dinov2/configs/train/fmri_vits.yaml     (new: fMRI YAML)
#   - 6 small edits to official files, each tagged "# FMRI CHANGE / WHY"
#
# Init weights: official DINOv2 ViT-S/14 reg4 ImageNet checkpoint,
# filtered to drop pos_embed + patch_embed.* (shape-mismatched against
# our 3D+1D embedding / (T_eff, N_spatial) token grid). The transformer
# blocks, CLS token, register tokens and final LayerNorm DO transfer.
#
# Usage (from the FAIR_official root):
#     cd $LAB_DIR/repos/FAIR_official
#     sbatch slurm_jobs/run_dinov2_fmri.sh           # default: 1 x H200
#     N_GPUS=2 sbatch slurm_jobs/run_dinov2_fmri.sh  # 2 x H200, edit #SBATCH below
# =====================================================================

#SBATCH --job-name=dinov2-fmri
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_jobs/logs/dinov2_fmri_%j.out
#SBATCH --error=slurm_jobs/logs/dinov2_fmri_%j.err
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

# To use 2 GPUs:
#   1) swap the #SBATCH line above for  #SBATCH --gres=gpu:h200:2
#   2) set N_GPUS=2 below (or export it before sbatch).

set -euo pipefail

N_GPUS=${N_GPUS:-1}

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
VENV_DIR="$LAB_DIR/torch_env"
CKPT_DIR="$LAB_DIR/checkpoints"
DINOV2_INIT="$CKPT_DIR/dinov2_vits14_reg4_fmri_init.pth"

CONFIG_FILE="dinov2/configs/train/fmri_vits.yaml"
RUN_NAME="dinov2_fmri_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OFFICIAL_DIR/outputs/$RUN_NAME"

export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TORCH_HOME="$LAB_DIR/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# distributed.enable() expects a rendezvous endpoint even for 1 GPU.
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29513
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$OUTPUT_DIR" "$CKPT_DIR"
mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs"

echo "============================================================"
echo "  DINOv2 OFFICIAL SSL on fMRI (HCP, ViT-S, H200 x $N_GPUS)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Repo:      $OFFICIAL_DIR"
echo "  Config:    $CONFIG_FILE"
echo "  Output:    $OUTPUT_DIR"
echo "  Init wts:  $DINOV2_INIT"
echo "============================================================"

source "$VENV_DIR/bin/activate"
cd "$OFFICIAL_DIR"

# ----- 1. Make sure xformers is installed (NestedTensorBlock needs it) -----
if ! python -c "import xformers" 2>/dev/null; then
    echo ""
    echo "  xformers missing -> pip install xformers"
    # No --index-url because the venv's torch dictates the right CUDA wheel;
    # let pip's resolver pick the wheel that matches the installed torch.
    pip install --no-input xformers
fi
python -c "import xformers; print(f'  xformers {xformers.__version__} OK')"

# ----- 2. Prepare filtered DINOv2 ImageNet init checkpoint (once) -----
if [ ! -f "$DINOV2_INIT" ]; then
    echo ""
    echo "  Preparing $DINOV2_INIT (one-shot)"
    python - <<PY
import torch
# Official DINOv2 ViT-S/14 reg4 ImageNet weights, via torch hub.
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
state = m.state_dict()
# Drop the two fMRI-incompatible groups:
#   - pos_embed: ImageNet (1, 257, 384), fMRI (1, T_eff*N_spat+1, 384) -> diff size
#   - patch_embed.*: ImageNet PatchEmbed (Conv2d 14x14x3), fMRI PatchEmbed3DPlus1D (Conv3d 9x9x9x1 + Conv1d)
# strict=False would still error on shape mismatch; filtering is the only way.
filtered = {
    k: v for k, v in state.items()
    if not (k.startswith("patch_embed") or k == "pos_embed")
}
torch.save({"model": filtered}, "$DINOV2_INIT")
kept = sorted(filtered.keys())
print(f"  saved {len(filtered)} keys (dropped {len(state) - len(filtered)})")
print(f"  kept top-level groups:",
      sorted({k.split('.')[0] for k in kept}))
PY
fi
ls -lh "$DINOV2_INIT"

# ----- 3. Echo the fMRI overrides for the run log -----
echo ""
echo "  Config snapshot (fMRI overrides):"
grep -E "^(  dataset_path|  batch_size_per_gpu|  fmri_|  arch|  patch_size|  num_register_tokens|  in_chans|  epochs|  base_lr|  centering|  pretrained_weights|  OFFICIAL_EPOCH_LENGTH)" \
    "$CONFIG_FILE" | sed 's/^/    /'
echo ""

# ----- 4. Launch -----
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=$MASTER_PORT \
    -m dinov2.train.train \
        --config-file "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Job finished: $(date)"
echo "  Final checkpoint dir: $OUTPUT_DIR"
echo "============================================================"
