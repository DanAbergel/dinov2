#!/bin/bash
# =====================================================================
# SLURM job — minimal test that wraps the NrgXnat bash script that
# DID work manually for danab95. Reads the password from
# ~/.xnat_password and pipes it into the bash script's `read -s` prompt
# so it runs unattended.
#
# This is a DIAGNOSTIC test, not the real download pipeline. If this
# succeeds inside a SLURM job, we know the auth + bash flow works in
# the cluster context. Then we wire it into the main task.
#
# Output: /tmp/oasis3_test/ on the compute node (one ZIP per scan)
#
# Usage:
#   echo 'your-password' > ~/.xnat_password
#   chmod 600 ~/.xnat_password
#   sbatch tasks/download_oasis3/test_bash.sh
# =====================================================================

#SBATCH --job-name=oasis3-test
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

export LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
export OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
export TASK_DIR="$OFFICIAL_DIR/tasks/download_oasis3"
export PYTHONUNBUFFERED=1

# Username — override with XNAT_USERNAME=other sbatch ...
XNAT_USERNAME="${XNAT_USERNAME:-danab95}"

mkdir -p "$TASK_DIR/logs"

LOG_OUT="$TASK_DIR/logs/test_bash.out"
LOG_ERR="$TASK_DIR/logs/test_bash.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/test_bash_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/test_bash_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

echo "============================================================"
echo "  OASIS-3 bash-wrapper TEST (no Python, no requests)"
echo "============================================================"
echo "  Job ID:   ${SLURM_JOB_ID:-(local)}"
echo "  Node:     $(hostname)"
echo "  Date:     $(date)"
echo "  Username: $XNAT_USERNAME"
echo "============================================================"

# ----- 1. Read password from ~/.xnat_password ------------------------
PW_FILE="$HOME/.xnat_password"
if [ ! -f "$PW_FILE" ]; then
    echo "ERROR: $PW_FILE not found." >&2
    echo "  Create it with: echo 'your-password' > $PW_FILE && chmod 600 $PW_FILE" >&2
    exit 2
fi
# Permission check
MODE=$(stat -c "%a" "$PW_FILE")
if [ "$MODE" != "600" ] && [ "$MODE" != "400" ]; then
    echo "ERROR: $PW_FILE has insecure perms ($MODE). Run: chmod 600 $PW_FILE" >&2
    exit 2
fi
XNAT_PASSWORD=$(cat "$PW_FILE")
PW_LEN=${#XNAT_PASSWORD}
PW_HEAD="${XNAT_PASSWORD:0:2}"
PW_TAIL="${XNAT_PASSWORD: -2}"
echo "  Password read from $PW_FILE :  '${PW_HEAD}...${PW_TAIL}' (len=$PW_LEN)"

# ----- 2. Clone NrgXnat oasis-scripts if missing ---------------------
WORK_DIR="/tmp/oasis_bash_test"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d oasis-scripts ]; then
    echo "Cloning NrgXnat/oasis-scripts ..."
    git clone --depth 1 https://github.com/NrgXnat/oasis-scripts.git
fi
cd oasis-scripts/download_scans
echo "Working dir: $(pwd)"

# ----- 3. Build a CSV with ONE known experiment_id -------------------
echo "OAS30001_MR_d0129" > test.csv
echo "  Test CSV created with 1 experiment_id:"
cat test.csv

# ----- 4. Run the bash script with the password piped via stdin -----
# The script uses `read -s -p ... PW`. read reads from stdin by default,
# so piping the password as one line works whether or not -s is set.
echo ""
echo "Running download_oasis_scans.sh with auto-injected password ..."
echo "----------------------------------------------------------------"
mkdir -p ./downloaded
echo "$XNAT_PASSWORD" | bash ./download_oasis_scans.sh test.csv ./downloaded "$XNAT_USERNAME"
SCRIPT_RC=$?

echo "----------------------------------------------------------------"
echo "Script returned: $SCRIPT_RC"

# ----- 5. Verify what was downloaded ---------------------------------
echo ""
echo "Listing ./downloaded :"
ls -la ./downloaded || true
find ./downloaded -name "*.nii.gz" -o -name "*.zip" 2>/dev/null | head -20 || true

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
