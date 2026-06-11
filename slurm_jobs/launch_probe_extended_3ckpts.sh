#!/bin/bash
# Launches probe_adni_extended.sh on the 3 final checkpoints (A / B / C).
# Run from the repo root, NOT submitted -- this is a plain bash launcher.
#
# Usage:
#   bash slurm_jobs/launch_probe_extended_3ckpts.sh

set -euo pipefail

OFFICIAL_DIR="/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official"
cd "$OFFICIAL_DIR"

# The 3 ablation runs, each at their final checkpoint.
# Names follow the convention from CHANGES_TO_DINOV2 / FREEZE_ABLATION_RESULTS:
#   A = full fine-tune       = dinov2_fmri_hcp_baseline
#   B = freeze last 3 blocks = dinov2_fmri_hcp_freeze_last3
#   C = freeze except input  = dinov2_fmri_hcp_freeze_fmri
RUNS=(
    "dinov2_fmri_hcp_baseline"
    "dinov2_fmri_hcp_freeze_last3"
    "dinov2_fmri_hcp_freeze_fmri"
)

for run in "${RUNS[@]}"; do
    # Pick the final checkpoint (largest iter number).
    ckpt=$(ls -t outputs/${run}/model_*.rank_0.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "WARN: no checkpoint for $run, skipping"
        continue
    fi
    echo "Submitting probe_adni_extended for: $ckpt"
    CHECKPOINT="$ckpt" sbatch slurm_jobs/probe_adni_extended.sh
done

echo ""
echo "Submitted. Watch logs:"
echo "  tail -f slurm_jobs/logs/probe_adni_extended_latest.out"
echo "  squeue -u \$USER"
