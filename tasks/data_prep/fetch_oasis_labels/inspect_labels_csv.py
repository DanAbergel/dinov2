"""Inspect ANY downloaded clinical/label CSV and show how it maps to our corpus.

Use this the moment you download a label file from a portal (OASIS-3 clinical
CDR, ADNI amyloid AV45, ADNIMERGE, CamCAN participants...). It prints the columns,
a few sample rows, and guesses which column holds the subject id + the label of
interest — enough for us to write the exact join + probe axis with no guessing.

Usage (on Moriah or locally):
    python tasks/data_prep/fetch_oasis_labels/inspect_labels_csv.py <file.csv>
    python .../inspect_labels_csv.py <file.csv> --grep cdr,dx,av45,amyloid
"""

import argparse
import csv
import sys
from collections import Counter


def sniff(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = list(reader)
    return rows, getattr(dialect, "delimiter", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--grep", default="cdr,dx,diagn,mmse,av45,amyloid,suvr,pib,"
                    "subject,id,ptid,oasisid,age,sex,gender,visit,day,date,label,group",
                    help="comma-separated substrings to flag as interesting columns")
    ap.add_argument("--rows", type=int, default=4, help="sample rows to print")
    args = ap.parse_args()

    rows, delim = sniff(args.csv)
    if not rows:
        sys.exit("empty file")
    header, data = rows[0], rows[1:]
    needles = [g.strip().lower() for g in args.grep.split(",") if g.strip()]

    print(f"file      : {args.csv}")
    print(f"delimiter : {delim!r}")
    print(f"columns   : {len(header)}   rows : {len(data)}\n")

    print("=== all columns (idx: name) ===")
    for i, c in enumerate(header):
        flag = "  <-- interesting" if any(n in c.lower() for n in needles) else ""
        print(f"  {i:3}: {c}{flag}")

    print(f"\n=== first {args.rows} rows (interesting columns only) ===")
    keep = [i for i, c in enumerate(header) if any(n in c.lower() for n in needles)]
    if keep:
        print("  " + " | ".join(header[i] for i in keep))
        for r in data[: args.rows]:
            print("  " + " | ".join(r[i] if i < len(r) else "" for i in keep))

    # For likely subject-id + likely label columns, show value distribution
    print("\n=== value distribution of flagged label-like columns ===")
    label_cols = [i for i, c in enumerate(header)
                  if any(n in c.lower() for n in ("cdr", "dx", "diagn", "amyloid",
                                                  "group", "label", "sex", "gender"))]
    for i in label_cols:
        vals = Counter((r[i] if i < len(r) else "") for r in data)
        top = ", ".join(f"{v!r}:{n}" for v, n in vals.most_common(8))
        print(f"  [{i}] {header[i]}: {top}")


if __name__ == "__main__":
    main()
