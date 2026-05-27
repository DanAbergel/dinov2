#!/bin/bash
# =====================================================================
# Submit ADNI+HCP probes on 3 checkpoints of the most recent training
# run, in parallel — idempotent and run-aware.
#
# Picks the OLDEST, MIDDLE, and NEWEST `model_*.rank_0.pth` available
# at submission time. If only 1 or 2 checkpoints exist, falls back to
# what's there.
#
# Run-aware behavior (added 2026-05-26):
#   - For each (probe_type, iter) target, checks if the corresponding
#     JSON in outputs/probes/ is FRESH (mtime > run dir mtime).
#       FRESH    -> skip (already done for this run).
#       STALE    -> move to outputs/probes/old_<run_name>/  and submit.
#       MISSING  -> submit.
#   This prevents collisions when probe filenames repeat across runs
#   (iter 2999 of a Mixed run vs iter 2999 of a previous HCP-only run)
#   and avoids re-running expensive probes you already have.
#
# Each probe is its own sbatch (parallel across SLURM); results land in
# outputs/probes/probe[_hcp]_iter<ITER>.json.
#
# Usage (from FAIR_official root):
#   bash slurm_jobs/probe_3ckpts.sh                    # auto-pick latest run
#   RUN_DIR=outputs/dinov2_fmri_<ts>  bash slurm_jobs/probe_3ckpts.sh
#   FORCE=1  bash slurm_jobs/probe_3ckpts.sh           # re-submit even fresh ones
#
# This is NOT itself a sbatch — it's a launcher (no SBATCH directives).
# =====================================================================

set -euo pipefail

OFFICIAL_DIR="/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official"
VENV_DIR="/sci/labs/arieljaffe/dan.abergel1/torch_env"
cd "$OFFICIAL_DIR"

# Activate venv so ckpt_iter() can import torch and read 'iteration' from
# .pth payloads (needed for model_final.rank_0.pth which has no numeric
# iter in its filename). Without this, python3 falls back to system python
# which lacks torch -> ckpt_iter silently returns empty -> the loop skips
# the final checkpoint.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PROBES_DIR="$OFFICIAL_DIR/outputs/probes"
mkdir -p "$PROBES_DIR"

# ----- 1. Resolve the run directory -----
if [ -z "${RUN_DIR:-}" ]; then
    RUN_DIR=$(ls -td outputs/dinov2_fmri_* 2>/dev/null | head -1 || true)
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: no run directory found. Set RUN_DIR=outputs/dinov2_fmri_<ts>."
    exit 2
fi
RUN_NAME=$(basename "$RUN_DIR")
RUN_DIR_MTIME=$(stat -c %Y "$RUN_DIR" 2>/dev/null || stat -f %m "$RUN_DIR")
OLD_BACKUP="$PROBES_DIR/old_${RUN_NAME}_predecessors"

echo "Run directory: $RUN_DIR"
echo "Run name:      $RUN_NAME"
echo "Run mtime:     $(date -d @$RUN_DIR_MTIME 2>/dev/null || date -r $RUN_DIR_MTIME)"
echo "Old backup:    $OLD_BACKUP  (used for any stale predecessor JSONs)"

# ----- 2. Enumerate checkpoints -----
mapfile -t CKPTS < <(ls -1 "$RUN_DIR"/model_*.rank_0.pth 2>/dev/null | sort)
N=${#CKPTS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no model_*.rank_0.pth in $RUN_DIR."
    exit 3
fi
echo ""
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
echo "Selected for probing:"
for c in "${SELECTED[@]}"; do echo "  $c"; done
echo ""

# ----- 4. Helper: is a probe JSON FRESH (produced for THIS run) ? -----
# Args: $1 = path to JSON
# Returns 0 (fresh), 1 (stale, exists but older than run), 2 (missing).
freshness() {
    local f="$1"
    if [ ! -f "$f" ]; then return 2; fi
    local m
    m=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f")
    if [ "$m" -gt "$RUN_DIR_MTIME" ]; then return 0; else return 1; fi
}

backup_stale() {
    local f="$1"
    mkdir -p "$OLD_BACKUP"
    mv "$f" "$OLD_BACKUP/$(basename "$f")"
    echo "    stale -> moved to $OLD_BACKUP/$(basename "$f")"
}

# ----- Helper: extract iter # from a checkpoint path -----
# Handles both model_<iter>.rank_0.pth (numeric) and model_final.rank_0.pth
# (read iter from inside the .pth via a tiny python call).
# Always emits a numeric line (or -1) on stdout so the caller never gets
# an empty string. Stderr goes to /dev/null so partial-import noise doesn't
# pollute the launcher log.
ckpt_iter() {
    local ckpt="$1"
    local raw
    raw=$(basename "$ckpt" | sed -E 's/^model_0*([0-9]+)\.rank_0\.pth$/\1/')
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        echo "$raw"
        return 0
    fi
    # model_final.rank_0.pth or similar: read iter from the .pth payload.
    local out
    out=$(python3 - <<PY 2>/dev/null || true
import torch
try:
    d = torch.load("$ckpt", map_location="cpu", weights_only=False)
    print(int(d.get("iteration", -1)))
except Exception:
    print(-1)
PY
)
    # Fallback: if python failed entirely (no torch, no python3), emit -1.
    if [[ -z "$out" ]]; then
        echo "-1"
    else
        echo "$out"
    fi
}

# ----- 5. Submit only what's missing or stale -----
SUBMITTED=0
SKIPPED=0
for CKPT in "${SELECTED[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing checkpoint: $CKPT"
    ITER=$(ckpt_iter "$CKPT")
    echo "  detected iter: '$ITER'"
    if ! [[ "$ITER" =~ ^[0-9]+$ ]] || [ "$ITER" -lt 0 ]; then
        echo "  [SKIP] could not determine iter (got '$ITER'). Check venv activation."
        continue
    fi
    ITER_PADDED=$(printf "%07d" "$ITER")

    for KIND in adni hcp; do
        if [ "$KIND" = "adni" ]; then
            JSON="$PROBES_DIR/probe_iter${ITER_PADDED}.json"
            SBATCH_SH="slurm_jobs/probe_adni.sh"
            JOB_NAME="probe-adni-i${ITER}"
        else
            JSON="$PROBES_DIR/probe_hcp_iter${ITER_PADDED}.json"
            SBATCH_SH="slurm_jobs/probe_hcp.sh"
            JOB_NAME="probe-hcp-i${ITER}"
        fi

        echo "[$KIND @ iter $ITER]"
        echo "    expected: $JSON"

        if [ -n "${FORCE:-}" ]; then
            echo "    FORCE=1 -> resubmitting unconditionally."
            if [ -f "$JSON" ]; then backup_stale "$JSON"; fi
        else
            # Capture freshness() exit code without losing it through `if ... fi`.
            # `set -e` requires `|| status=$?` so non-zero returns don't kill the script.
            status=0
            freshness "$JSON" || status=$?
            case "$status" in
                0) echo "    FRESH (mtime > run mtime) -> skip."
                   SKIPPED=$((SKIPPED + 1))
                   continue
                   ;;
                1) echo "    STALE (older than run) -> move + resubmit."
                   backup_stale "$JSON"
                   ;;
                2) echo "    MISSING -> submit."
                   ;;
                *) echo "    UNKNOWN freshness status ($status) -> submit."
                   ;;
            esac
        fi

        sbatch --export=ALL,CHECKPOINT="$CKPT" \
               --job-name="$JOB_NAME" \
               "$SBATCH_SH"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================================"
echo "  Submitted: $SUBMITTED   Skipped (already fresh): $SKIPPED"
echo "============================================================"
echo "Monitor with:"
echo "   squeue -u \$USER"
echo "Results land in:"
echo "   $PROBES_DIR/probe{,_hcp}_iter*.json"
echo ""
echo "Tail latest log:"
echo "   tail -f $OFFICIAL_DIR/slurm_jobs/logs/probe_adni_latest.out"
echo "   tail -f $OFFICIAL_DIR/slurm_jobs/logs/probe_hcp_latest.out"
echo ""
echo "To force-resubmit even fresh probes: FORCE=1 bash $0"
