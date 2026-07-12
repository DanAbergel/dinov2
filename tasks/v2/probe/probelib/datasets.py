"""One class per cohort: its comparison tasks {name: label column} and samples().

Each sample is a dict {path, subject, tr, labels{col: value}}. A dataset only
knows how to read its own manifest / phenotype format; everything downstream
(embedding, split, probe) is generic. REGISTRY maps a --dataset name to a class.

In-corpus cohorts (ADNI/ABIDE/HCP/OASIS) read the shared corpus_manifest.csv;
external cohorts (ADHD/COBRE/UCLA) glob one .pt per subject (subclass _Glob).
"""

import csv
from pathlib import Path

import numpy as np

from dinov2.data.fmri_data import LAB_ROOT, DEFAULT_MANIFEST

LAB = Path(LAB_ROOT)
REPO = Path(__file__).resolve().parents[4]      # probelib -> probe -> v2 -> tasks -> repo
MANIFEST = LAB / DEFAULT_MANIFEST


def _f(v):
    """Parse to float, or None (NaN / non-numeric)."""
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except (TypeError, ValueError):
        return None


def _median_bin(samples, field, key):
    """Add a binary label `key` = 1 if `field` >= global median else 0 (NaN if missing)."""
    vals = [_f(s["labels"].get(field)) for s in samples]
    med = np.median([v for v in vals if v is not None]) if any(vals) else None
    for s, v in zip(samples, vals):
        s["labels"][key] = float("nan") if v is None or med is None else float(v >= med)


# ---------- in-corpus cohorts (70:30 via subject_split.json) ----------

class ADNI:
    name = "ADNI"
    tasks = {"NC_vs_MCI": "nc_vs_mci", "AD_vs_HC": "ad_vs_hc", "Amyloid": "amyloid_positive"}
    DIR = LAB / "ADNI_data" / "downsampled"

    def samples(self):
        clin = {}
        p = next((c for c in (self.DIR / "adni_clinical.csv",
                              REPO / "data" / "adni_clinical.csv") if c.exists()), None)
        if p:
            clin = {r["subject_id"]: r for r in csv.DictReader(open(p))}
        out = []
        for r in csv.DictReader(open(self.DIR / "adni_manifest.csv")):
            sid, iid = r["subject_id"], r["image_id"]
            path = self.DIR / sid / f"{iid}.pt"
            if not path.exists():
                continue
            cdr = _f(r.get("Global CDR"))                # proxy: NC=0, MCI=0.5, AD>=1
            lab = {"nc_vs_mci": (0.0 if cdr == 0 else 1.0 if cdr == 0.5 else float("nan")),
                   "ad_vs_hc": (0.0 if cdr == 0 else 1.0 if cdr and cdr >= 1 else float("nan")),
                   "amyloid_positive": float("nan")}
            c = clin.get(sid)                            # real DX + amyloid overrides proxy
            if c:
                for k in lab:
                    if _f(c.get(k)) is not None:
                        lab[k] = _f(c[k])
            out.append({"path": path, "subject": sid, "tr": 3.0, "labels": lab})
        return out


class ABIDE:
    name = "ABIDE"
    tasks = {"Autism": "autism", "Age": "age_bin", "Sex": "sex_bin"}

    def samples(self):
        pheno = {r["FILE_ID"]: r for r in
                 csv.DictReader(open(LAB / "ABIDE_data" / "abide_phenotypic.csv"))}
        out = []
        for r in csv.DictReader(open(MANIFEST)):
            if r["dataset"] != "ABIDE":
                continue
            ph = pheno.get(r["subject_id"].replace("_downsampled", ""))
            dx = _f(ph.get("DX_GROUP")) if ph else None  # 1=autism 2=control
            if dx is None:
                continue
            sex = _f(ph.get("SEX"))                       # 1=male 2=female
            out.append({"path": Path(r["path"]), "subject": r["subject_id"], "tr": float(r["tr"]),
                        "labels": {"autism": float(dx == 1),
                                   "sex_bin": float(sex == 1) if sex is not None else float("nan"),
                                   "AGE_AT_SCAN": ph.get("AGE_AT_SCAN")}})
        _median_bin(out, "AGE_AT_SCAN", "age_bin")
        return out


class HCP:
    name = "HCP"
    tasks = {"Sex": "sex_bin", "Age": "age_bin"}

    def samples(self):
        meta = {str(r["Subject"]).strip(): r for r in
                csv.DictReader(open(REPO / "data" / "HCP_YA_subjects.csv"))}
        out = []
        for r in csv.DictReader(open(MANIFEST)):
            if r["dataset"] != "HCP":
                continue
            sid = r["subject_id"]
            m = meta.get(sid) or meta.get("".join(c for c in sid if c.isdigit())[:6])
            if not m:
                continue
            g = (m.get("Gender") or "").strip()
            out.append({"path": Path(r["path"]), "subject": sid, "tr": float(r["tr"]),
                        "labels": {"sex_bin": 1.0 if g == "M" else 0.0 if g == "F" else float("nan"),
                                   "Age_in_Yrs": m.get("Age_in_Yrs")}})
        _median_bin(out, "Age_in_Yrs", "age_bin")
        return out


class OASIS:
    name = "OASIS"
    tasks = {"AD_Conversion": "ad_conversion"}

    def samples(self):
        lab_file = LAB / "OASIS3_data" / "oasis_labels.csv"
        labels = ({r["subject_id"]: _f(r.get("ad_conversion")) for r in csv.DictReader(open(lab_file))}
                  if lab_file.exists() else {})
        out = []
        for r in csv.DictReader(open(MANIFEST)):
            if r["dataset"] != "OASIS":
                continue
            sid = r["subject_id"]
            key = next((k for k in (sid, sid.split("_")[0]) if k in labels), None)
            out.append({"path": Path(r["path"]), "subject": sid, "tr": float(r["tr"]),
                        "labels": {"ad_conversion": labels.get(key, float("nan"))}})
        return out


# ---------- external cohorts (never pretrained on; random 70:30) ----------

class _Glob:
    """Downstream-only cohort = one .pt per subject in DIR + a per-subject label lookup."""
    DIR = None

    def _label(self, subject):
        raise NotImplementedError

    def samples(self):
        out = []
        for p in sorted(self.DIR.glob("*_downsampled.pt")):
            subj = p.name.replace("_downsampled.pt", "")
            lab = self._label(subj)
            if lab is not None:
                out.append({"path": p, "subject": subj, "tr": self.TR, "labels": lab})
        return out


class ADHD(_Glob):
    name = "ADHD"
    TR = 2.0
    DIR = LAB / "ADHD200_data" / "downsampled"
    tasks = {"ADHD": "adhd"}

    def __init__(self):
        rows = list(csv.reader(open(LAB / "ADHD200_data" / "adhd200_phenotypic.csv")))
        h = [c.strip().lower() for c in rows[0]]
        si = next((i for i, c in enumerate(h) if "scandir" in c or "subject" in c), 0)
        di = next((i for i, c in enumerate(h) if "dx" in c), None)
        self.dx = {}
        for r in rows[1:]:
            d = _f(r[di]) if di is not None and di < len(r) else None
            digits = "".join(c for c in r[si] if c.isdigit())
            if digits and d is not None:
                self.dx[int(digits)] = 0.0 if d == 0 else 1.0

    def _label(self, subj):
        digits = "".join(c for c in subj.split("_session")[0] if c.isdigit())
        v = self.dx.get(int(digits)) if digits else None
        return None if v is None else {"adhd": v}


class COBRE(_Glob):
    name = "COBRE"
    TR = 2.0
    DIR = LAB / "COBRE_data" / "downsampled"
    tasks = {"Schizophrenia": "sz"}

    def __init__(self):
        f = LAB / "COBRE_data" / "cobre_labels.csv"
        self.sz = {r["subject_id"].strip(): _f(r.get("sz")) for r in csv.DictReader(open(f))} if f.exists() else {}

    def _label(self, subj):
        v = self.sz.get(subj)
        return None if v is None else {"sz": v}


class UCLA(_Glob):
    name = "UCLA"
    TR = 2.0
    DIR = LAB / "UCLA_data" / "downsampled"
    tasks = {"Schizophrenia": "schizophrenia", "ADHD": "adhd"}

    def __init__(self):
        f = LAB / "UCLA_data" / "ucla_participants.tsv"
        self.dx = {r["participant_id"].strip(): (r.get("diagnosis") or "").strip().upper()
                   for r in csv.DictReader(open(f), delimiter="\t")} if f.exists() else {}

    def _label(self, subj):
        d = self.dx.get(subj)
        if d is None:
            return None
        return {"schizophrenia": 1.0 if d == "SCHZ" else 0.0 if d == "CONTROL" else float("nan"),
                "adhd": 1.0 if d == "ADHD" else 0.0 if d == "CONTROL" else float("nan")}


REGISTRY = {c.name: c for c in [ADNI, ABIDE, HCP, OASIS, ADHD, COBRE, UCLA]}
