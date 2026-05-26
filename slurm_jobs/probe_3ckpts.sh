#!/bin/bash
# =====================================================================
# Submit ADNI+HCP probes on 3 checkpoints of the most recent training
# run, in parallel.
#
# Picks the OLDEST, MIDDLE, and NEWEST `model_*.rank_0.pth` available
# at submission time. If only one checkpoint exists, runs just that one;
# if 2, runs oldest + newest; if 3+, runs 3 well-spread ones.
#
# Each probe is its own sbatch (parallel across SLURM); results land in
# outputs/probes/probe[_hcp]_iter<ITER>.json.
#
# Usage (from FAIR_official root):
#   bash slurm_jobs/probe_3ckpts.sh                   # auto-pick latest run
#   RUN_DIR=outputs/dinov2_fmri_<ts>  bash slurm_jobs/probe_3ckpts.sh
#
# This is NOT itself a sbatch — it's a launcher script (no SBATCH dirs).
# =====================================================================

set -euo pipefail

OFFICIAL_DIR="/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official"
cd "$OFFICIAL_DIR"

# ----- 1. Resolve the run directory -----
if [ -z "${RUN_DIR:-}" ]; then
    RUN_DIR=$(ls -td outputs/dinov2_fmri_* 2>/dev/null | head -1 || true)
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: no run directory found. Set RUN_DIR=outputs/dinov2_fmri_<ts>."
    exit 2
fi
echo "Run directory: $RUN_DIR"

# ----- 2. Enumerate checkpoints -----
mapfile -t CKPTS < <(ls -1 "$RUN_DIR"/model_*.rank_0.pth 2>/dev/null | sort)
N=${#CKPTS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no model_*.rank_0.pth in $RUN_DIR."
    exit 3
fi
echo "Found $N checkpoint(s):"
for c in "${CKPTS[@]}"; do echo "  $c"; done

# ----- 3. Pick 3 spread-out checkpoints (oldest, middle, newest) -----
if [ "$N" -eq 1 ]; then
    SELECTED=("${CKPTS[0]}")
elif [ "$N" -eq 2 ]; then
    SELECTED=("${CKPTS[0]}" "${CKPTS[1]}")
else
    MID_IDX=$((N / 2))
    SELECTED=("${CKPTS[0]}" "${CKPTS[$MID_IDX]}" "${CKPTS[$((N - 1))]}")
fi

echo ""
echo "Probing on:"
for c in "${SELECTED[@]}"; do echo "  $c"; done
echo ""

# ----- 4. Submit ADNI + HCP probes for each checkpoint -----
for CKPT in "${SELECTED[@]}"; do
    ITER=$(basename "$CKPT" | sed -E 's/^model_0*([0-9]+)\.rank_0\.pth$/\1/')
    echo "Submitting ADNI probe for iter $ITER ..."
    sbatch --export=ALL,CHECKPOINT="$CKPT" \
           --job-name="probe-adni-i${ITER}" \
           slurm_jobs/probe_adni.sh
    echo "Submitting HCP  probe for iter $ITER ..."
    sbatch --export=ALL,CHECKPOINT="$CKPT" \
           --job-name="probe-hcp-i${ITER}" \
           slurm_jobs/probe_hcp.sh
done

echo ""
echo "All probes submitted. Monitor with:"
echo "   squeue -u \$USER"
echo "Results land in:"
echo "   $OFFICIAL_DIR/outputs/probes/probe{,_hcp}_iter*.json"
echo ""
echo "Tail latest log: tail -f $OFFICIAL_DIR/slurm_jobs/logs/probe_adni_latest.out"
