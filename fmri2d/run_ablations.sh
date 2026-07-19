#!/bin/bash
# Submit a GRID of prototype x temperature ablations on the multi-frame + local=global data.
# Each combo = one sbatch job (fmri2d/ablation.sh) that TRAINS then PROBES (subject-level 5-fold CV).
# Prereq: brain2d_frames already extracted (run test_frames.sh once).
#
# Usage (run on the LOGIN node, not sbatch):
#   bash fmri2d/run_ablations.sh
#   PROTOS_LIST="2048 4096" TEMP_LIST="0.01:0.04:25" bash fmri2d/run_ablations.sh   # custom grid
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p fmri2d/logs                              # so each job's %x.out can be created at start

PROTOS_LIST="${PROTOS_LIST:-1024 2048 4096}"
# temperature schedules as "warmupTemp:teacherTemp:warmupEpochs"
TEMP_LIST="${TEMP_LIST:-0.01:0.04:25 0.01:0.02:25}"
CV="${CV:-5}"

n=0
for P in $PROTOS_LIST; do
  for T in $TEMP_LIST; do
    IFS=: read -r WTT TTv WE <<< "$T"
    RUN="abl_p${P}_${WTT}-${TTv}_w${WE}"
    echo "submit  $RUN   (protos=$P  temp=$WTT->$TTv  warmup=$WE ep)"
    RUN="$RUN" PROTOS="$P" WARMUP_TT="$WTT" TT="$TTv" WARMUP_EPOCHS="$WE" CV="$CV" \
      sbatch -A arieljaffe --job-name="$RUN" fmri2d/ablation.sh
    n=$((n + 1))
  done
done
echo ""
echo "submitted $n jobs."
echo "watch:    squeue -u \$USER"
echo "results:  grep -A2 'CV accuracy' fmri2d/logs/abl_*.out"
