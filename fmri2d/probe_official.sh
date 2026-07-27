#!/bin/bash
# OFFICIAL DINOv2 linear-probe protocol (dinov2/eval/linear.py) reproduced on our brain PNGs,
# predicting sex (Gender), with a SUBJECT-LEVEL train/val split (no leakage).
#
# Prereqs: brain PNGs extracted (extract_hcp.sh) AND a DINOv2 run trained on them.
# Launch:
#   RUN_DIR=$LAB/runs/brain/<run> PROBE_OPTS="dino.head_n_prototypes=2048 ibot.head_n_prototypes=2048" \
#     sbatch -A arieljaffe fmri2d/probe_official.sh
# Watch:  tail -f fmri2d/logs/probe_official_<run>.out
#
#SBATCH --job-name=brain-probe-official
#SBATCH --account=arieljaffe
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official
#SBATCH --output=fmri2d/probe_official_slurm.out
#SBATCH --error=fmri2d/probe_official_slurm.err
set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
CONFIG="$LAB_DIR/repos/FAIR_official/dinov2/configs/train/imagenette_vits.yaml"
FEATURES_ROOT="${FEATURES_ROOT:-$LAB_DIR/brain2d}"          # where extract_hcp.sh wrote the PNGs
RUN_DIR="${RUN_DIR:?set RUN_DIR to the brain-DINOv2 training output dir (has the checkpoint)}"

# per-run logs so concurrent probes don't overwrite each other (out/err separate)
mkdir -p fmri2d/logs
LOGBASE="fmri2d/logs/probe_official_$(basename "$RUN_DIR")"
exec > "$LOGBASE.out" 2> "$LOGBASE.err"

export TMPDIR="$LAB_DIR/tmp"; export XDG_CACHE_HOME="$LAB_DIR/cache"; export HOME="$LAB_DIR"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"; export TORCHINDUCTOR_CACHE_DIR="$LAB_DIR/cache/inductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# HCP sex/age labels live on the fMRI branch — pull the CSV out (no network, no commit).
LABELS="$TMPDIR/HCP_YA_subjects.csv"
git show origin/fmri-multi-source:data/HCP_YA_subjects.csv > "$LABELS"
echo "labels: $(wc -l < "$LABELS") rows -> $LABELS"

source "$LAB_DIR/torch_env/bin/activate"
export PYTHONPATH="$LAB_DIR/repos/FAIR_official:${PYTHONPATH:-}"

srun python fmri2d/probe_official.py \
    --config-file "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --features-root "$FEATURES_ROOT" \
    --labels-csv "$LABELS" \
    --label-col "${LABEL_COL:-Gender}" \
    ${VAL_FRAC:+--val-frac "$VAL_FRAC"} \
    ${NO_AUG:+--no-aug} \
    ${PROBE_OPTS:-}
