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

# Extract labels with python (no requests needed, just stdlib json).
# Keep only real subjects matching OAS3 followed by digits (filters out
# metadata folders like '0AS_data_files').
SUBJECTS_TXT="$WORK_DIR/subjects.txt"
python3 -c "
import json, re
d = json.load(open('$SUBJECTS_JSON'))
pat = re.compile(r'^OAS3\d+\$')
for r in d['ResultSet']['Result']:
    if pat.match(r['label']):
        print(r['label'])
" | sort > "$SUBJECTS_TXT"
N_SUBJ=$(wc -l < "$SUBJECTS_TXT")
echo "  found $N_SUBJ real OAS3 subjects"

# Skip subjects already downloaded (a dir OAS3XXXX_MR_* already exists in
# raw_nifti). Lets a 2nd job resume where a previous run stopped, without
# re-querying/re-downloading the ~726 already on disk.
RAW_DIR="$OASIS3_DIR/raw_nifti"
if [ -d "$RAW_DIR" ]; then
    ls "$RAW_DIR" 2>/dev/null | sed -E 's/_MR_.*$//' | sort -u > "$WORK_DIR/already.txt"
    N_ALREADY=$(wc -l < "$WORK_DIR/already.txt")
    comm -23 "$SUBJECTS_TXT" "$WORK_DIR/already.txt" > "$SUBJECTS_TXT.todo"
    mv "$SUBJECTS_TXT.todo" "$SUBJECTS_TXT"
    echo "  skipping $N_ALREADY already-downloaded subjects"
    echo "  remaining to process: $(wc -l < "$SUBJECTS_TXT")"
fi

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
# Read the subject list on FD 3 so curls inside the loop (which read stdin)
# can't consume it. This was corrupting subject IDs (e.g. 'OAS30005' -> 'AS30005').
while IFS= read -r SUBJ <&3; do
    i=$((i+1))
    EXP_JSON="$WORK_DIR/${SUBJ}_exp.json"
    curl -f -k -s </dev/null --cookie "$COOKIE_JAR" \
        "$XNAT_HOST/data/archive/projects/$PROJECT/subjects/$SUBJ/experiments?format=json&xsiType=xnat:mrSessionData" \
        > "$EXP_JSON" || { echo "  [$i/$TOTAL] $SUBJ: list-exp failed"; continue; }

    # Get all MR sessions sorted earliest-first.
    SESSIONS=$(python3 -c "
import json, re
d = json.load(open('$EXP_JSON'))
items = d['ResultSet']['Result']
def days(s):
    m = re.search(r'_d(\d+)\$', s.get('label',''))
    return int(m.group(1)) if m else 9999999
for s in sorted(items, key=days):
    print(s['label'], s['ID'])
")
    rm -f "$EXP_JSON"
    if [ -z "$SESSIONS" ]; then
        echo "  [$i/$TOTAL] $SUBJ: no MR sessions"
        continue
    fi

    # Walk sessions earliest-first; take the first that HAS a bold scan.
    CHOSEN=""
    while read -r SESS_LABEL SESS_ID; do
        [ -z "$SESS_LABEL" ] && continue
        SCANS_JSON="$WORK_DIR/${SESS_LABEL}_scans.json"
        if ! curl -f -k -s </dev/null --cookie "$COOKIE_JAR" \
            "$XNAT_HOST/data/archive/experiments/$SESS_ID/scans?format=json" \
            > "$SCANS_JSON" 2>/dev/null; then
            rm -f "$SCANS_JSON"; continue
        fi
        HAS_BOLD=$(python3 -c "
import json, sys
try:
    d = json.load(open('$SCANS_JSON'))
except Exception:
    print('no'); sys.exit()
scans = d['ResultSet']['Result']
ok = any('bold' in (s.get('type','')+s.get('series_description','')).lower()
         for s in scans)
print('yes' if ok else 'no')
")
        rm -f "$SCANS_JSON"
        if [ "$HAS_BOLD" = "yes" ]; then
            CHOSEN="$SESS_LABEL"
            break
        fi
    done <<< "$SESSIONS"

    if [ -z "$CHOSEN" ]; then
        echo "  [$i/$TOTAL] $SUBJ: no session with a bold scan"
        continue
    fi
    echo "$CHOSEN" >> "$CSV_FILE"
    echo "  [$i/$TOTAL] $SUBJ -> $CHOSEN"
done 3< "$SUBJECTS_TXT"

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
