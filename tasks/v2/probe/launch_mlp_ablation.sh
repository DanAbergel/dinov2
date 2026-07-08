#!/bin/bash
# =====================================================================
# Fan out the MLP-architecture ablation: ONE separate SLURM job per
# (dataset x arch), all running in PARALLEL. Run this on the LOGIN node.
#
#   RUN      (default base)
#   DATASETS (default: ADNI ABIDE HCP ADHD COBRE)   # OASIS skipped (no labels)
#   ARCHS    (default: 128 256 256,128 512,256 512,256,128)
#
# Each job writes probe_<run>_<ds>_mlp-<arch>.json and does NO git.
# After ALL jobs finish, commit + push ONCE (json results AND logs):
#   git add tasks/v2/probe/json_results/probe_*_mlp-*.json tasks/v2/probe/logs/probe1_*_mlp-*
#   git commit -m mlp && git push
#
# Usage:
#   bash tasks/v2/probe/launch_mlp_ablation.sh
#   DATASETS="ADNI ABIDE HCP" ARCHS="256 512,256" bash tasks/v2/probe/launch_mlp_ablation.sh
# =====================================================================
set -euo pipefail
TASK_DIR="/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official/tasks/v2/probe"

RUN="${RUN:-base}"
DATASETS="${DATASETS:-ADNI ABIDE HCP ADHD COBRE}"
ARCHS="${ARCHS:-128 256 256,128 512,256 512,256,128}"

n=0
for DS in $DATASETS; do
    for A in $ARCHS; do
        tag="${A//,/x}"
        # pass vars as an ENV PREFIX (+ --export=ALL), NOT in the --export list:
        # sbatch's --export list is COMMA-separated, so MLP_ARCH=512,256 there would
        # be truncated to 512. The env prefix preserves commas intact.
        RUN="$RUN" DATASET="$DS" HEAD=mlp AGG=mean MLP_ARCH="$A" \
            sbatch -A arieljaffe --job-name="mlp-${DS}-${tag}" --export=ALL \
            "$TASK_DIR/probe_one.sh"
        n=$((n+1))
    done
done
echo "submitted $n MLP-ablation jobs (run=$RUN)"
echo "watch:  squeue -u \$USER -n $(echo $DATASETS | tr ' ' ,)  # or squeue -u \$USER | grep mlp-"
echo "when done, commit ONCE (json + logs):"
echo "  cd $(dirname "$TASK_DIR")/../.. && git add tasks/v2/probe/json_results/probe_${RUN}_*_mlp-*.json tasks/v2/probe/logs/probe1_${RUN}_*_mlp-* && git commit -m 'mlp ablation' && git push"
