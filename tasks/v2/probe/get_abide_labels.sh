#!/bin/bash
# =====================================================================
# Download ABIDE I phenotypic labels -> ABIDE_data/abide_phenotypic.csv
# Columns we use: FILE_ID (matches scan name), DX_GROUP (1=autism, 2=control),
# SEX (1=male, 2=female), AGE_AT_SCAN, SITE_ID.
#
# Public (no auth). Run on the GATEWAY (needs internet):
#   bash tasks/v2/probe/get_abide_labels.sh
# =====================================================================
set -euo pipefail
LAB="/sci/labs/arieljaffe/dan.abergel1"
OUT="$LAB/ABIDE_data/abide_phenotypic.csv"
URL="https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"

mkdir -p "$(dirname "$OUT")"
echo "Downloading ABIDE phenotypic CSV ..."
curl -fsSL -o "$OUT" "$URL"
echo "Saved $OUT  ($(wc -l < "$OUT") rows)"
echo "Autism/Control counts (DX_GROUP):"
python3 -c "
import csv, collections
c=collections.Counter(r['DX_GROUP'] for r in csv.DictReader(open('$OUT')))
print('  DX_GROUP (1=autism, 2=control):', dict(c))
"
