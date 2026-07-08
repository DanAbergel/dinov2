"""Leakage-free 70:30 probe for a trained fMRI run.

Four pieces, nothing else:
  cls_token()    -> one embedding per scan (mean CLS over sliding 270-windows)
  Probe(nn)      -> the head: hidden=() linear, hidden=(256,128) MLP; forward -> 1 logit
  train_probe()  -> THE training function: Adam + BCE + backprop on the FROZEN
                    embeddings, return test AUC/Acc/F1. hidden=() linear else MLP.
  Dataset        -> one class per cohort: its comparison tasks {name: label} + samples()
  run()          -> samples -> embeddings (cached) -> 70:30 -> train_probe per task -> JSON

70:30 split: in-pretraining cohorts (ADNI/ABIDE/HCP/OASIS) use subject_split.json
(the 30% test was held out of pretraining -> no leakage); downstream-only cohorts
(ADHD/COBRE/UCLA, never pretrained on) get a deterministic random 70:30.

Usage:
    python probe.py --run-dir .../runs/v2/base --dataset ADNI --out out.json
    python probe.py --run-dir .../runs/v2/base --dataset UCLA --mlp 256,128 --out out.json
"""

import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dinov2.models import build_model_from_cfg
from dinov2.data.fmri_data import (_load_mmap, _temporal_resample, _zscore_per_frame,
                                   TARGET_TR, LAB_ROOT, DEFAULT_SPLIT, DEFAULT_MANIFEST)

LAB = Path(LAB_ROOT)
REPO = Path(__file__).resolve().parents[3]
MANIFEST = LAB / DEFAULT_MANIFEST
T_FIXED = 270
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _f(v):
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except (TypeError, ValueError):
        return None


# ---- encoder: one CLS embedding per scan ----

def load_teacher(run_dir, checkpoint):
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    _s, teacher, _dim = build_model_from_cfg(cfg)
    ckpt = torch.load(Path(run_dir) / checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    tb = {k.replace("_fsdp_wrapped_module.", "")[len("teacher.backbone."):]: v
          for k, v in state.items()
          if k.replace("_fsdp_wrapped_module.", "").startswith("teacher.backbone.")}
    teacher.load_state_dict(tb, strict=False)
    return teacher.to(DEVICE).eval()


@torch.no_grad()
def cls_token(teacher, path, native_tr):
    """Mean CLS token over sliding native windows, each resampled to 270 @ 0.72s."""
    win = max(1, round(T_FIXED * TARGET_TR / native_tr))
    stride = max(1, win // 2)
    scan = _load_mmap(path).float()
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                       # (T,1,X,Y,Z)
    embs = []
    for s in range(0, max(scan.shape[0] - win + 1, 1), stride):
        clip = _zscore_per_frame(_temporal_resample(scan[s:s + win].clone(), T_FIXED))
        out = teacher(clip.unsqueeze(0).to(DEVICE), is_training=True)
        embs.append(out["x_norm_clstoken"].squeeze(0).float().cpu())
    return torch.stack(embs).mean(0).numpy()


# ---- the probe head (PyTorch) + THE training function ----

class Probe(nn.Module):
    """A probe head on top of the frozen CLS embedding.
    hidden=() -> a single Linear (logistic regression);
    hidden=(256, 128) -> an MLP with those hidden layers (ReLU). Output = 1 logit."""

    def __init__(self, d_in, hidden=()):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)                 # (B,) logits


def train_probe(Xtr, ytr, Xte, yte, hidden=(), epochs=200, lr=1e-3, weight_decay=1e-4):
    """Train a Probe head with Adam + BCE + backprop, return test AUC/Acc/F1.
    hidden=() -> linear; hidden=(256,128) -> MLP. Encoder stays frozen (we train
    only this head on the pre-extracted embeddings)."""
    # standardize on TRAIN stats
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEVICE)
    xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)

    model = Probe(Xtr.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # class balance: pos_weight = #neg / #pos
    npos = float(ytr.sum())
    pos_weight = torch.tensor([(len(ytr) - npos) / max(npos, 1.0)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr_t)              # forward
        loss.backward()                                # backprop
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(xte)).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    return {"auc": float(roc_auc_score(yte, proba)), "acc": float(accuracy_score(yte, pred)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "n_train": int(len(ytr)), "n_test": int(len(yte)), "pos_test": int(yte.sum())}


# ---- one class per dataset: comparison tasks {name: label col} + samples() ----

def _median_bin(samples, field, key):
    vals = [_f(s["labels"].get(field)) for s in samples]
    med = np.median([v for v in vals if v is not None]) if any(vals) else None
    for s, v in zip(samples, vals):
        s["labels"][key] = float("nan") if v is None or med is None else float(v >= med)


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


class _Glob:
    """Downstream-only cohort = one .pt per subject in DIR + a label lookup."""
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


# ---- 70:30 split + runner ----

def split_masks(name, subjects):
    """70% train / 30% test by subject. Pretrained cohorts use subject_split.json
    (30% held out of pretraining); others get a deterministic random 70:30."""
    sp = json.loads((LAB / DEFAULT_SPLIT).read_text())["datasets"]
    if name in sp:
        m = {s: k for k, subs in sp[name].items() for s in subs}
        return (np.array([m.get(s) == "train" for s in subjects]),
                np.array([m.get(s) in ("val", "test") for s in subjects]))
    uniq = sorted(set(subjects))
    np.random.RandomState(0).shuffle(uniq)
    train = set(uniq[:int(0.7 * len(uniq))])
    tr = np.array([s in train for s in subjects])
    return tr, ~tr


def run(dataset, teacher, run_dir, checkpoint, mlp, out):
    S = dataset.samples()
    print(f"{dataset.name}: {len(S)} scans", flush=True)
    cache = Path(run_dir) / f"emb_{dataset.name}_{checkpoint.replace('.', '_')}.npz"
    if cache.exists() and np.load(cache)["X"].shape[0] == len(S):
        X = np.load(cache)["X"]
    else:
        X = np.stack([cls_token(teacher, s["path"], s["tr"]) for s in S])
        np.savez(cache, X=X)
    subj = np.array([s["subject"] for s in S])
    tr_all, te_all = split_masks(dataset.name, subj)
    results = {}
    for task, col in dataset.tasks.items():
        y = np.array([_f(s["labels"].get(col)) for s in S], dtype=float)
        ok = ~np.isnan(y)
        tr, te = tr_all & ok, te_all & ok
        if tr.sum() < 10 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            results[task] = None
            print(f"  {task:16} (skipped)", flush=True)
            continue
        r = train_probe(X[tr], y[tr], X[te], y[te], hidden=mlp or ())
        results[task] = r
        print(f"  {task:16} AUC {r['auc']:.3f}  Acc {r['acc']:.3f}  F1 {r['f1']:.3f}  n_te={r['n_test']}", flush=True)
    payload = {"run": Path(run_dir).name, "dataset": dataset.name, "mlp": mlp, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"Saved {out}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--mlp", default=None, help="MLP hidden sizes, e.g. '256,128' (default: linear)")
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mlp = tuple(int(w) for w in args.mlp.split(",")) if args.mlp else None
    teacher = load_teacher(args.run_dir, args.checkpoint)
    run(REGISTRY[args.dataset](), teacher, args.run_dir, args.checkpoint, mlp, args.out)


if __name__ == "__main__":
    main()
