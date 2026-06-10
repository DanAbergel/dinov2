"""Inspect ADNI label distributions.

For every clinical column referenced in `probe_labels.ADNI_LABELS` (plus the
raw CDR / degradation columns used to build them), report:
  - total number of scans in the tensor
  - number with non-null value
  - number missing
  - distinct values and their counts + percentages (for discrete / binary cols)
  - min / max / mean / std + a 5-bin histogram (for continuous cols)

Restriction: we only look at scans whose index appears in
`index_to_name.json` -- i.e. scans that exist in the ADNI 4D tensor we
actually train/probe on. Entries in `imageID_to_labels.json` that have no
matching scan are ignored.

Usage:
    python scripts/inspect_adni_labels.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_labels import (
    ADNI_INDEX_JSON, ADNI_LABELS_JSON, ADNI_LABELS, load_adni_labels,
)


# Discrete-by-default: enumerate all observed values. Everything else is
# treated as continuous (histogram).
DISCRETE_COLS = {
    "Sex", "Sex_Binary",
    "CDR", "CDR_Binary",
    "degradation_binary_1year",
    "degradation_binary_2years",
    "degradation_binary_3years",
    "Group", "DX", "Diagnosis", "ResearchGroup",
}

CONTINUOUS_COLS = {
    "Age", "MMSE Total Score", "MMSE",
}


def value_counts_report(series: pd.Series, name: str):
    n_total = len(series)
    nn = series.dropna()
    n_valid = len(nn)
    n_missing = n_total - n_valid

    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    print(f"  total scans     : {n_total}")
    print(f"  valid (non-NaN) : {n_valid}  ({100*n_valid/n_total:5.1f}%)")
    print(f"  missing         : {n_missing}  ({100*n_missing/n_total:5.1f}%)")

    if n_valid == 0:
        print("  (no values)")
        return

    distinct = nn.unique()
    discrete = (name in DISCRETE_COLS) or (
        len(distinct) <= 12 and name not in CONTINUOUS_COLS
    )

    if discrete:
        counts = Counter(nn)
        print(f"  distinct values : {len(distinct)}")
        print(f"  {'value':<20} {'count':>8} {'% of valid':>12} {'% of total':>12}")
        for val, c in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print(f"    {str(val):<18} {c:>8} {100*c/n_valid:>11.2f}% {100*c/n_total:>11.2f}%")
    else:
        try:
            arr = nn.astype(float).values
        except (ValueError, TypeError):
            print(f"  (mixed types, distinct={distinct[:10]}...)")
            return
        print(f"  min / max       : {arr.min():.3f}  /  {arr.max():.3f}")
        print(f"  mean / std      : {arr.mean():.3f}  /  {arr.std():.3f}")
        # 5-bin histogram, equal-width
        bins = np.linspace(arr.min(), arr.max(), 6)
        hist, edges = np.histogram(arr, bins=bins)
        print(f"  histogram (5 equal-width bins):")
        for i, c in enumerate(hist):
            print(f"    [{edges[i]:7.2f} .. {edges[i+1]:7.2f})  {c:>6}  "
                  f"{100*c/n_valid:5.1f}%")


def main():
    print(f"Loading ADNI labels:")
    print(f"  index : {ADNI_INDEX_JSON}")
    print(f"  labels: {ADNI_LABELS_JSON}")

    labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)

    # ----- Top-level dataset size -----
    n_scans = len(labels_df)
    n_subjects = labels_df["subject_id"].nunique()
    scans_per_subj = labels_df.groupby("subject_id").size()
    print(f"\n{'='*72}")
    print(f"  DATASET SIZE")
    print(f"{'='*72}")
    print(f"  Total scans                  : {n_scans}")
    print(f"  Total unique subjects        : {n_subjects}")
    print(f"  Scans per subject  -- mean   : {scans_per_subj.mean():.2f}")
    print(f"                     -- median : {int(scans_per_subj.median())}")
    print(f"                     -- min/max: {scans_per_subj.min()} / {scans_per_subj.max()}")
    print(f"  Subjects with >=2 scans      : {(scans_per_subj >= 2).sum()}")
    print(f"  Subjects with  1  scan       : {(scans_per_subj == 1).sum()}")

    # Per-label subject counts (since some scans drop out for NaN labels,
    # the effective number of subjects per probe is smaller than the total).
    print(f"\n  Effective subject count per probe label:")
    print(f"  {'label':<32} {'scans':>7} {'subjects':>10}")
    for name, lcfg in ADNI_LABELS.items():
        col = lcfg["column"]
        if col not in labels_df.columns:
            print(f"    {name:<30} (column {col!r} not in DataFrame)")
            continue
        mask = labels_df[col].notna()
        n_s_scans = int(mask.sum())
        n_s_subj = labels_df.loc[mask, "subject_id"].nunique()
        print(f"    {name:<30} {n_s_scans:>7} {n_s_subj:>10}")

    print(f"\nLabels DataFrame: {labels_df.shape}")
    print(f"All columns       : {list(labels_df.columns)}")

    # Columns referenced by ADNI_LABELS (the ones the probes actually use):
    probe_cols = sorted({cfg["column"] for cfg in ADNI_LABELS.values()})
    print(f"\nColumns USED by probes ({len(probe_cols)}):")
    for c in probe_cols:
        present = "OK " if c in labels_df.columns else "MISS"
        print(f"  [{present}]  {c}")

    # Inspect each probe column.
    for col in probe_cols:
        if col in labels_df.columns:
            value_counts_report(labels_df[col], col)

    # Bonus: inspect the raw (non-binary) CDR + Diagnosis columns if they
    # exist, so we know how the binary was built.
    BONUS_COLS = ["CDR", "CDR_GLOBAL", "Group", "DX", "Diagnosis",
                  "ResearchGroup", "diagnosis", "dx_bl"]
    extras = [c for c in BONUS_COLS if c in labels_df.columns and c not in probe_cols]
    if extras:
        print(f"\n\n{'#'*72}")
        print(f"#  BONUS: raw clinical columns (not used by probes, for context)")
        print(f"{'#'*72}")
        for col in extras:
            value_counts_report(labels_df[col], col)


if __name__ == "__main__":
    main()
