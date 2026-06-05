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
#SBATCH --mem=256G
#SBATCH --time=72:00:00
# SLURM's own output is discarded; we redirect to a per-run timestamped log
# file in-script below so each launch keeps its own log (matching its
# outputs/<RUN_NAME> directory). A "latest" symlink points at the most
# recent run for `tail -f` convenience.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

# To use 2 GPUs:
#   1) swap the #SBATCH line above for  #SBATCH --gres=gpu:h200:2
#   2) set N_GPUS=2 below (or export it before sbatch).

set -euo pipefail

N_GPUS=${N_GPUS:-1}

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export VENV_DIR="$LAB_DIR/torch_env"
export CKPT_DIR="$LAB_DIR/checkpoints"
export DINOV2_INIT="$CKPT_DIR/dinov2_vits14_reg4_fmri_init.pth"

# Use CONFIG_FILE from env if set (e.g. for freeze ablations), else default
# to the canonical Mixed-T=140 config. Override with:
#   CONFIG_FILE=dinov2/configs/train/fmri_vits_hcp_freeze_fmri.yaml sbatch ...
export CONFIG_FILE="${CONFIG_FILE:-dinov2/configs/train/fmri_vits.yaml}"

# RUN_NAME = the config file name (= a real, indicative description). Same
# config => same RUN_NAME => same log file => relaunch overwrites the previous
# attempt of THIS config. Different config => different RUN_NAME => never
# collides with another config's logs/outputs.
#   fmri_vits.yaml                       -> RUN_NAME "dinov2_fmri_default"
#   fmri_vits_hcp_freeze_fmri.yaml       -> RUN_NAME "dinov2_fmri_hcp_freeze_fmri"
#   fmri_vits_hcp_freeze_last3.yaml      -> RUN_NAME "dinov2_fmri_hcp_freeze_last3"
CONFIG_TAG=$(basename "$CONFIG_FILE" .yaml | sed 's/^fmri_vits_*//')
[ -z "$CONFIG_TAG" ] && CONFIG_TAG="default"
export RUN_NAME="dinov2_fmri_${CONFIG_TAG}"
export OUTPUT_DIR="$OFFICIAL_DIR/outputs/$RUN_NAME"

# ----- Per-run timestamped log (preserves history of all runs) -----
# Log filename matches the output directory's RUN_NAME so it's trivial to
# match a log to its checkpoints. `latest` symlink for `tail -f`.
mkdir -p "$OFFICIAL_DIR/slurm_jobs/logs"
LOG_OUT="$OFFICIAL_DIR/slurm_jobs/logs/${RUN_NAME}.out"
LOG_ERR="$OFFICIAL_DIR/slurm_jobs/logs/${RUN_NAME}.err"
ln -sf "${RUN_NAME}.out" "$OFFICIAL_DIR/slurm_jobs/logs/dinov2_fmri_latest.out"
ln -sf "${RUN_NAME}.err" "$OFFICIAL_DIR/slurm_jobs/logs/dinov2_fmri_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export XDG_CACHE_HOME="$LAB_DIR/cache"
export TORCH_HOME="$LAB_DIR/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# distributed.enable() expects a rendezvous endpoint even for 1 GPU.
export MASTER_ADDR="127.0.0.1"
# Derive a unique MASTER_PORT per job (modulo SLURM_JOB_ID) so two
# trainings landing on the same node (parallel ablations) don't both
# try port 29513 and collide with EADDRINUSE. The %1000 keeps the port
# in [29000, 30000). MASTER_PORT can still be overridden via env.
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-$RANDOM} % 1000))}"
# Verbose diagnostics — printed to the .out / .err files so we can see
# the real traceback when torchrun loses it.
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
# Print every bash command before execution (with line numbers).
set -x
PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
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

# ----- 1. Install missing official dinov2 deps (idempotent) -----
# Only the runtime imports actually used by dinov2/{models,data,train,loss,
# layers,utils,fsdp,distributed,logging}: torch is already there, we add
# the missing pieces. We do NOT pin torch / torchvision (the venv already
# has them at a known-working version) and we skip submitit / cuml / mmcv
# / mmseg / ftfy which are only needed for eval / segmentation / submitit
# launcher — none of which we use.
declare -A REQUIRED_PKGS=(
    [xformers]=xformers
    [fvcore]=fvcore
    [iopath]=iopath
    [omegaconf]=omegaconf
    [einops]=einops
)
for mod in "${!REQUIRED_PKGS[@]}"; do
    if ! python -c "import $mod" 2>/dev/null; then
        echo ""
        echo "  $mod missing -> pip install ${REQUIRED_PKGS[$mod]}"
        pip install --no-input "${REQUIRED_PKGS[$mod]}"
    fi
done
python -c "
import xformers, fvcore, iopath, omegaconf, einops
print(f'  xformers  {xformers.__version__}')
print(f'  fvcore    {fvcore.__version__ if hasattr(fvcore, \"__version__\") else \"OK\"}')
print(f'  iopath    OK')
print(f'  omegaconf {omegaconf.__version__}')
print(f'  einops    {einops.__version__}')
"

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

# ----- 4. Pre-flight: verify the model + config load cleanly on CPU -----
# This catches schema / import errors before we burn a GPU slot. If this
# fails we exit early with the python traceback in the log.
echo ""
echo "  Pre-flight: build_model_from_cfg on CPU"
python - <<'PY'
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
import os, sys, traceback
sys.path.insert(0, os.environ.get("OFFICIAL_DIR", "."))
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
cfg = OmegaConf.merge(
    OmegaConf.create(dinov2_default_config),
    OmegaConf.load(os.environ["CONFIG_FILE"]),
)
print(f"    cfg.train.dataset_path = {cfg.train.dataset_path}")
print(f"    cfg.student.fmri_mode  = {cfg.student.fmri_mode}")
try:
    from dinov2.models import build_model_from_cfg
    student, teacher, embed_dim = build_model_from_cfg(cfg)
    print(f"    student total params = {sum(p.numel() for p in student.parameters()):,}")
    print(f"    pre-flight OK")
except Exception:
    print("    pre-flight FAILED:")
    traceback.print_exc()
    sys.exit(2)
PY

# ----- 5. Launch (with per-rank logs in slurm_jobs/logs/torchrun_<jobid>/) -----
RANK_LOG_DIR="$OFFICIAL_DIR/slurm_jobs/logs/torchrun_${SLURM_JOB_ID:-local}"
mkdir -p "$RANK_LOG_DIR"
# Hide SLURM_JOB_ID from the children so dinov2.distributed.enable()
# stops trying _set_from_slurm_env() (which needs SLURM_NTASKS/SLURM_PROCID/
# SLURM_LOCALID — only set by `srun`, not by plain `sbatch`). With it
# unset, _TorchDistributedEnvironment falls back to _set_from_preset_env()
# which reads MASTER_ADDR/PORT/RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE
# — exactly what torchrun injects into every spawned worker.
env -u SLURM_JOB_ID -u SLURM_JOB_NUM_NODES -u SLURM_JOB_NODELIST \
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_port=$MASTER_PORT \
    --redirects 3 \
    --tee 3 \
    --log-dir "$RANK_LOG_DIR" \
    -m dinov2.train.train \
        --config-file "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Job finished: $(date)"
echo "  Final checkpoint dir: $OUTPUT_DIR"
echo "============================================================"
