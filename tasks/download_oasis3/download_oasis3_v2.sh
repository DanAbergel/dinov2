#!/bin/bash
# =====================================================================
# SLURM job — OASIS-3 baseline rs-fMRI downloader.
#
# Pure-bash approach (no Python in the auth/query path). Wraps the
# NrgXnat official download script that is known to work on NITRC-IR.
#
# Pipeline:
#   1. Auth + list subjects via curl (uses ~/.xnat_password)
#   2. For each subject, query earliest MR session ID
#   3. Generate a CSV of experiment IDs
#   4. Run download_oasis_scans.sh on that CSV
#   5. Run a small Python script to downsample all downloaded .nii.gz
#
# Usage:
#   sbatch tasks/download_oasis3/download_oasis3_v2.sh
#   LIMIT=5 sbatch tasks/download_oasis3/download_oasis3_v2.sh   # test 5 subj
# =====================================================================

#SBATCH --job-name=oasis3-v2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
OFFICIAL_DIR="$LAB_DIR/repos/FAIR_official"
TASK_DIR="$OFFICIAL_DIR/tasks/download_oasis3"
OASIS3_DIR="$LAB_DIR/OASIS3_data"
VENV_DIR="$LAB_DIR/torch_env"
WORK_DIR="$LAB_DIR/tmp/oasis3_work"

XNAT_HOST="https://www.nitrc.org/ir"
PROJECT="OASIS3"
XNAT_USERNAME="${XNAT_USERNAME:-danab95}"

mkdir -p "$TASK_DIR/logs" "$OASIS3_DIR/downsampled" "$OASIS3_DIR/raw_nifti" "$WORK_DIR"

LOG_OUT="$TASK_DIR/logs/download_oasis3_v2.out"
LOG_ERR="$TASK_DIR/logs/download_oasis3_v2.err"
ln -sf "$(basename "$LOG_OUT")" "$TASK_DIR/logs/download_oasis3_v2_latest.out"
ln -sf "$(basename "$LOG_ERR")" "$TASK_DIR/logs/download_oasis3_v2_latest.err"
exec >"$LOG_OUT" 2>"$LOG_ERR"

echo "============================================================"
echo "  OASIS-3 v2 download (pure-bash auth + Python downsample)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-(local)}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Username:  $XNAT_USERNAME"
echo "============================================================"

# ----- 0. Password -----------------------------------------------------
PW_FILE="$HOME/.xnat_password"
[ -f "$PW_FILE" ] || { echo "ERROR: $PW_FILE missing" >&2; exit 2; }
XNAT_PASSWORD_RAW=$(cat "$PW_FILE")
echo "  password read (len=${#XNAT_PASSWORD_RAW})"

# CRITICAL: NITRC-IR expects the password URL-encoded in the Basic Auth
# header (this is what the NrgXnat script does and why it works). Reproduce
# their escape_chars_for_URL function. Without this, a '#' (or other special
# char) in the password causes a 401.
escape_chars_for_URL() {
    echo "${1}" | sed -e 's/%/%25/g;' | sed -e 's/ /%20/g; s/</%3C/g; s/>/%3E/g; s/#/%23/g; s/+/%2B/g; s/{/%7B/g; s/}/%7D/g; s/|/%7C/g; s/\\/%5C/g; s/\^/%5E/g; s/~/%7E/g; s/\[/%5B/g; s/\]/%5D/g; s/`/%60/g; s/;/%3B/g; s/?/%3F/g; s/:/%3A/g; s/@/%40/g; s/=/%3D/g; s/&/%26/g; s/\$/%24/g'
}
# Username is also lowercased by the NrgXnat script.
XNAT_USERNAME=$(echo "$XNAT_USERNAME" | tr 'A-Z' 'a-z')
XNAT_PASSWORD=$(escape_chars_for_URL "$XNAT_PASSWORD_RAW")
echo "  password URL-encoded (len=${#XNAT_PASSWORD})"

# ----- 1. Clone NrgXnat scripts if missing ----------------------------
SCRIPTS_DIR="$WORK_DIR/oasis-scripts"
if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "Cloning NrgXnat/oasis-scripts ..."
    git clone --depth 1 https://github.com/NrgXnat/oasis-scripts.git "$SCRIPTS_DIR"
fi

# ----- 2. Authenticate via JSESSION (exactly like NrgXnat startSession).
#         With the URL-encoded password this now succeeds.
COOKIE_JAR="$WORK_DIR/cookies.jar"
echo "Authenticating (JSESSION, URL-encoded password) ..."
if ! curl -f -k -s -u "${XNAT_USERNAME}:${XNAT_PASSWORD}" \
        --cookie-jar "$COOKIE_JAR" \
        "$XNAT_HOST/data/JSESSION" > /dev/null; then
    echo "ERROR: JSESSION auth failed. Bad username/password?" >&2
    exit 3
fi
echo "  auth OK (cookie jar: $COOKIE_JAR)"

# ----- 3. List subjects (using the session cookie) --------------------
SUBJECTS_JSON="$WORK_DIR/subjects.json"
echo "Listing subjects ..."
curl -f -k -s --cookie "$COOKIE_JAR" \
    "$XNAT_HOST/data/archive/projects/$PROJECT/subjects?format=json" \
    > "$SUBJECTS_JSON"

# Extract labels with python (no requests needed, just stdlib json)
SUBJECTS_TXT="$WORK_DIR/subjects.txt"
python3 -c "
import json, sys
d = json.load(open('$SUBJECTS_JSON'))
for r in d['ResultSet']['Result']:
    print(r['label'])
" | sort > "$SUBJECTS_TXT"
N_SUBJ=$(wc -l < "$SUBJECTS_TXT")
echo "  found $N_SUBJ subjects"

# Limit if requested
if [ -n "${LIMIT:-}" ]; then
    head -n "$LIMIT" "$SUBJECTS_TXT" > "$SUBJECTS_TXT.lim"
    mv "$SUBJECTS_TXT.lim" "$SUBJECTS_TXT"
    echo "  limited to first $LIMIT subjects"
fi

# ----- 4. Find earliest MR session ID per subject ---------------------
CSV_FILE="$WORK_DIR/experiments.csv"
echo "experiment_id" > "$CSV_FILE"

i=0
TOTAL=$(wc -l < "$SUBJECTS_TXT")
while read SUBJ; do
    i=$((i+1))
    EXP_JSON="$WORK_DIR/${SUBJ}_exp.json"
    curl -f -k -s --cookie "$COOKIE_JAR" \
        "$XNAT_HOST/data/archive/projects/$PROJECT/subjects/$SUBJ/experiments?format=json&xsiType=xnat:mrSessionData" \
        > "$EXP_JSON" || { echo "  [$i/$TOTAL] $SUBJ: list-exp failed"; continue; }

    # Find earliest session (smallest _dXXXX)
    EARLIEST=$(python3 -c "
import json, re
d = json.load(open('$EXP_JSON'))
items = d['ResultSet']['Result']
def days(s):
    m = re.search(r'_d(\d+)\$', s.get('label',''))
    return int(m.group(1)) if m else 9999999
items.sort(key=days)
print(items[0]['label'] if items else '')
")
    if [ -z "$EARLIEST" ]; then
        echo "  [$i/$TOTAL] $SUBJ: no MR sessions"
        rm -f "$EXP_JSON"
        continue
    fi
    echo "$EARLIEST" >> "$CSV_FILE"
    echo "  [$i/$TOTAL] $SUBJ -> $EARLIEST"
    rm -f "$EXP_JSON"
done < "$SUBJECTS_TXT"

N_EXP=$(( $(wc -l < "$CSV_FILE") - 1 ))
echo "Total experiments to download: $N_EXP"

# ----- 5. Download all selected experiments via the official script ----
RAW_DIR="$OASIS3_DIR/raw_nifti"
echo "Downloading ALL scan types for each experiment to $RAW_DIR ..."
# The official bash script reads password via `read -s` and URL-encodes it
# ITSELF, so we pipe the RAW (un-encoded) password to avoid double-encoding.
# We only download the 'bold' scan type to skip anat/dwi/etc. and save time.
echo "$XNAT_PASSWORD_RAW" | bash "$SCRIPTS_DIR/download_scans/download_oasis_scans.sh" \
    "$CSV_FILE" "$RAW_DIR" "$XNAT_USERNAME" bold

# ----- 6. Downsample with Python --------------------------------------
source "$VENV_DIR/bin/activate"
for pkg in nibabel; do
    python -c "import $pkg" 2>/dev/null || pip install --no-input "$pkg"
done

echo "Downsampling all downloaded rs-fMRI .nii.gz ..."
python3 "$TASK_DIR/downsample_oasis3.py" \
    --input-dir "$RAW_DIR" \
    --output-dir "$OASIS3_DIR/downsampled"

# Cleanup raw if you want — for now keep them.
echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $OASIS3_DIR/downsampled"
echo "============================================================"
