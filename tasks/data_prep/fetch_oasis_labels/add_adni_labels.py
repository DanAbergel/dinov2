"""Join ADNI clinical labels (real DX + amyloid) onto our ADNI subjects.

Brain-JEPA's ADNI tasks are NC-vs-MCI (real clinical diagnosis) and Amyloid beta
+/-. Our Sagi manifest only has a CDR proxy and no amyloid, so we join a file
you download from LONI IDA:

  ADNIMERGE.csv           -> has PTID, VISCODE, DX (CN/MCI/Dementia), AV45 (SUVR)
  (or) UCBERKELEYAV45.csv -> has RID, VISCODE2, SUMMARYSUVR_WHOLECEREBNORM[_1.11CUTOFF]

We match by PTID (== our subject_id, e.g. 002_S_0413) at the BASELINE visit and
write <ADNI_DIR>/adni_clinical.csv with:
    subject_id, dx, nc_vs_mci, ad_vs_hc, av45_suvr, amyloid_positive
probe.py picks it up automatically (adds the real-DX + Amyloid axes).

Usage (once you've downloaded the file from LONI):
    python tasks/data_prep/fetch_oasis_labels/add_adni_labels.py --adnimerge /path/ADNIMERGE.csv
    python .../add_adni_labels.py --av45 /path/UCBERKELEYAV45.csv       # amyloid only
"""

import argparse
import csv
from pathlib import Path

LAB = Path("/sci/labs/arieljaffe/dan.abergel1")
ADNI_DIR = LAB / "ADNI_data" / "downsampled"
AV45_CUTOFF = 1.11


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(d, *keys):
    for k in keys:
        if k in d and d[k] not in ("", None):
            return d[k]
    return ""


def parse_adnimerge(path):
    """PTID -> {dx, av45} at the baseline visit (fallback: any visit)."""
    out = {}
    for r in csv.DictReader(open(path)):
        ptid = _first(r, "PTID", "ptid")
        if not ptid:
            continue
        visc = _first(r, "VISCODE", "VISCODE2").lower()
        dx = _first(r, "DX", "DX_bl")
        av45 = _f(_first(r, "AV45"))
        rec = out.setdefault(ptid, {})
        # prefer baseline; else keep first non-empty
        if visc in ("bl", "sc", "scmri") or "dx" not in rec:
            if dx:
                rec["dx"] = dx
            if av45 is not None:
                rec["av45"] = av45
    return out


def parse_av45(path):
    """UCBERKELEYAV45: no PTID (RID only) -> we can only add amyloid keyed by RID.
    We store rid->suvr and let the ADNIMERGE path resolve PTID; if only this file
    is given we key amyloid by RID and match against subject_id trailing digits."""
    out = {}
    for r in csv.DictReader(open(path)):
        rid = _first(r, "RID")
        if not rid:
            continue
        suvr = _f(_first(r, "SUMMARYSUVR_WHOLECEREBNORM"))
        pos = _first(r, "SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF")
        rec = out.setdefault(rid, {})
        if suvr is not None and "av45" not in rec:
            rec["av45"] = suvr
        if pos not in ("", None) and "pos" not in rec:
            rec["pos"] = _f(pos)
    return out


def dx_to_labels(dx):
    d = (dx or "").upper()
    nc = d in ("CN", "NL", "SMC")
    mci = "MCI" in d
    ad = d in ("DEMENTIA", "AD")
    nc_vs_mci = 0.0 if nc else 1.0 if mci else float("nan")
    ad_vs_hc = 0.0 if nc else 1.0 if ad else float("nan")
    return nc_vs_mci, ad_vs_hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adnimerge", help="ADNIMERGE.csv (PTID, DX, AV45)")
    ap.add_argument("--av45", help="UCBERKELEYAV45.csv (RID, SUVR) — amyloid only")
    ap.add_argument("--out", default=str(ADNI_DIR / "adni_clinical.csv"))
    args = ap.parse_args()
    if not args.adnimerge and not args.av45:
        raise SystemExit("give --adnimerge and/or --av45")

    merge = parse_adnimerge(args.adnimerge) if args.adnimerge else {}
    av45_by_rid = parse_av45(args.av45) if args.av45 else {}

    # subjects come from our manifest
    manifest = ADNI_DIR / "adni_manifest.csv"
    subs = sorted({r["subject_id"] for r in csv.DictReader(open(manifest))})

    rows, n_dx, n_amy = [], 0, 0
    for sid in subs:
        rec = dict(merge.get(sid, {}))
        # amyloid from av45 file by RID (trailing digits of PTID 002_S_0413 -> 413)
        if av45_by_rid:
            rid = sid.split("_")[-1].lstrip("0") or "0"
            a = av45_by_rid.get(rid, {})
            rec.setdefault("av45", a.get("av45"))
            if a.get("pos") is not None:
                rec["amyloid_positive"] = a["pos"]
        dx = rec.get("dx", "")
        nc_vs_mci, ad_vs_hc = dx_to_labels(dx)
        suvr = rec.get("av45")
        amy = rec.get("amyloid_positive")
        if amy is None and suvr is not None:
            amy = 1.0 if suvr > AV45_CUTOFF else 0.0
        if dx:
            n_dx += 1
        if amy is not None:
            n_amy += 1
        rows.append({"subject_id": sid, "dx": dx,
                     "nc_vs_mci": nc_vs_mci, "ad_vs_hc": ad_vs_hc,
                     "av45_suvr": suvr if suvr is not None else "",
                     "amyloid_positive": amy if amy is not None else ""})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "dx", "nc_vs_mci",
                                          "ad_vs_hc", "av45_suvr", "amyloid_positive"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    print(f"  {len(rows)} subjects | real DX for {n_dx} | amyloid for {n_amy}")


if __name__ == "__main__":
    main()
