"""Derive OASIS-3 'AD Conversion' labels from the ADRC Clinical Data spreadsheet.

Brain-JEPA's OASIS-3 task = AD Conversion: a subject who is non-demented at
baseline (CDR 0) and later reaches dementia (CDR >= threshold) is a converter (1);
one who stays CDR 0 is a non-converter (0); already-impaired-at-baseline -> excluded.

Download the CSV from www.oasis-brains.org (project OASIS3 -> table "ADRC Clinical
Data" -> Options -> Spreadsheet). Columns are auto-detected (subject / cdr / an
ordering column such as days-from-entry or Age); override with flags if needed.

Writes <OASIS_DIR>/oasis_labels.csv : subject_id, ad_conversion, cdr_current, n_visits
probe.py picks it up (build_table_oasis -> AD_Conversion axis).

Usage:
    python tasks/data_prep/fetch_oasis_labels/derive_oasis_adconv.py --clinical /path/ADRC_ClinicalData.csv
    # if auto-detect fails, inspect first then override:
    python .../derive_oasis_adconv.py --clinical f.csv --subject-col Subject --cdr-col cdr --order-col ageAtEntry
"""

import argparse
import csv
import re
from pathlib import Path

LAB = Path("/sci/labs/arieljaffe/dan.abergel1")
OASIS_DIR = LAB / "OASIS3_data"
OAS_RE = re.compile(r"OAS3\d+")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _detect(header, rows, kind):
    """Guess a column by name/content. kind in {subject, cdr, order}."""
    low = [h.lower() for h in header]
    if kind == "subject":
        for i, h in enumerate(low):
            if h in ("subject", "oasisid", "oasis_id", "adrc_adrcclinicaldata id", "id"):
                return i
        # else: the column whose values look like OAS3xxxx
        for i in range(len(header)):
            if any(OAS_RE.search(r[i]) for r in rows[:20] if i < len(r)):
                return i
    if kind == "cdr":
        for i, h in enumerate(low):
            if h == "cdr" or ("cdr" in h and "sob" not in h and "sum" not in h):
                return i
    if kind == "order":
        for key in ("days", "day", "ageatentry", "age_at_entry", "age", "visit"):
            for i, h in enumerate(low):
                if key in h:
                    return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--threshold", type=float, default=1.0, help="CDR for 'AD' (default 1.0)")
    ap.add_argument("--subject-col")
    ap.add_argument("--cdr-col")
    ap.add_argument("--order-col")
    ap.add_argument("--out", default=str(OASIS_DIR / "oasis_labels.csv"))
    args = ap.parse_args()

    with open(args.clinical, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]

    def col(flag, kind):
        if flag:
            return header.index(flag)
        return _detect(header, data, kind)

    si, ci, oi = col(args.subject_col, "subject"), col(args.cdr_col, "cdr"), col(args.order_col, "order")
    if si is None or ci is None:
        raise SystemExit(f"could not detect subject/cdr columns. header={header}\n"
                         f"-> re-run with --subject-col X --cdr-col Y (see inspect_labels_csv.py)")
    print(f"columns: subject={header[si]!r}  cdr={header[ci]!r}  "
          f"order={header[oi] if oi is not None else '(row order)'!r}")

    # collect per-subject (order, cdr) trajectory
    traj = {}
    for k, r in enumerate(data):
        m = OAS_RE.search(r[si]) if si < len(r) else None
        sid = m.group(0) if m else (r[si] if si < len(r) else "")
        cdr = _f(r[ci]) if ci < len(r) else None
        if not sid or cdr is None:
            continue
        order = _f(r[oi]) if (oi is not None and oi < len(r)) else k
        traj.setdefault(sid, []).append((order if order is not None else k, cdr))

    rows_out, n_conv, n_stable = [], 0, 0
    for sid, seq in traj.items():
        seq.sort(key=lambda t: t[0])
        cdrs = [c for _, c in seq]
        base, cur = cdrs[0], cdrs[-1]
        later_max = max(cdrs[1:]) if len(cdrs) > 1 else base
        if base == 0 and len(cdrs) > 1:
            adconv = 1.0 if later_max >= args.threshold else 0.0
        else:
            adconv = float("nan")            # already impaired at baseline -> exclude
        n_conv += adconv == 1.0
        n_stable += adconv == 0.0
        rows_out.append({"subject_id": sid, "ad_conversion": adconv,
                         "cdr_current": cur, "n_visits": len(cdrs)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "ad_conversion", "cdr_current", "n_visits"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {args.out}")
    print(f"  {len(rows_out)} subjects | converters={n_conv} | stable-non-demented={n_stable} "
          f"| excluded (impaired at baseline / single visit)={len(rows_out)-n_conv-n_stable}")


if __name__ == "__main__":
    main()
