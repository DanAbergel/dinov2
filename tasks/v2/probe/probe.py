"""Leakage-free probe for a trained fMRI run — OOP, one class per dataset.

Pipeline (see the four blocks below):
  1. Encoder          — load the frozen teacher, extract ONE CLS embedding per scan
  2. metrics          — one function per metric (auroc / accuracy / f1 / pearson)
  3. evaluation       — one function per task type
                        (binary_split / binary_kfold / multiclass_kfold / regression_split)
  4. Dataset + Task   — one Dataset subclass per cohort; each declares its samples,
                        its Tasks (comparison labels + metric + SOTA ref) and eval mode

A Dataset yields `samples` (path / subject / split / native TR / labels). run_probe
extracts (and caches) embeddings, then evaluates every Task with the protocol that
matches (task kind × dataset eval mode). Output JSON is unchanged, so the existing
runners (run_probe_ablations.sh, probe_one.sh, ...) keep working.

Adding a dataset = subclass Dataset, list its Tasks, implement samples(). Nothing else.

Usage:
    python probe.py --run-dir .../runs/v2/base --dataset ADNI
    python probe.py --run-dir .../runs/v2/base --dataset UCLA --head mlp --agg mean_std
"""

import argparse
import csv
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from dinov2.models import build_model_from_cfg
from dinov2.data.fmri_data import (
    _load_mmap, _temporal_resample, _zscore_per_frame,
    TARGET_TR, LAB_ROOT, DEFAULT_SPLIT, DEFAULT_MANIFEST,
)

LAB = Path(LAB_ROOT)
REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_MANIFEST = LAB / DEFAULT_MANIFEST
T_FIXED = 270
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# hyper-parameter grids (chosen by subject-aware CV on TRAIN only)
C_GRID = [0.01, 0.1, 1.0, 10.0]                       # LogReg inverse-reg
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]                 # MLP L2
MLP_ARCHS = [(128,), (256,), (256, 128), (512, 256), (512, 256, 128)]
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]   # Ridge (regression)


def _to_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


# =====================================================================
# 1. ENCODER — one CLS embedding per scan
# =====================================================================

class Encoder:
    """A run's FROZEN teacher backbone. `cls_token` turns one 4D scan into a
    single vector (mean, or mean+std, of the CLS over sliding 270-frame windows)."""

    def __init__(self, run_dir, checkpoint="model_final.rank_0.pth"):
        self.run_dir = Path(run_dir)
        self.checkpoint = checkpoint
        cfg = OmegaConf.load(self.run_dir / "config.yaml")
        _student, teacher, embed_dim = build_model_from_cfg(cfg)
        ckpt = torch.load(self.run_dir / checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        tb = {}
        for k, v in state.items():
            kk = k.replace("_fsdp_wrapped_module.", "")
            if kk.startswith("teacher.backbone."):
                tb[kk[len("teacher.backbone."):]] = v
        if not tb:
            raise RuntimeError(f"No teacher.backbone.* keys. Sample: {list(state)[:8]}")
        missing, unexpected = teacher.load_state_dict(tb, strict=False)
        print(f"teacher: loaded {len(tb)} keys  missing={len(missing)} "
              f"unexpected={len(unexpected)}  embed_dim={embed_dim}", flush=True)
        self.teacher = teacher.to(DEVICE).eval()
        self.embed_dim = embed_dim

    @torch.no_grad()
    def cls_token(self, path, native_tr, agg="mean"):
        """CLS aggregated over sliding native windows (each spanning T_FIXED*0.72s,
        resampled to T_FIXED @ 0.72s). agg='mean' -> 384-d; 'mean_std' -> 768-d."""
        win = max(1, round(T_FIXED * TARGET_TR / native_tr))
        stride = max(1, win // 2)
        scan = _load_mmap(path).float()                  # (T, X, Y, Z)
        if scan.ndim == 4:
            scan = scan.unsqueeze(1)                      # (T, 1, X, Y, Z)
        T = scan.shape[0]
        embs = []
        for s in range(0, max(T - win + 1, 1), stride):
            clip = scan[s:s + win].clone()
            clip = _temporal_resample(clip, T_FIXED)      # -> 270 @ 0.72s
            clip = _zscore_per_frame(clip)
            x = clip.unsqueeze(0).to(DEVICE)              # (1, T_FIXED, 1, X, Y, Z)
            out = self.teacher(x, is_training=True)
            embs.append(out["x_norm_clstoken"].squeeze(0).float().cpu())
        embs = torch.stack(embs)                          # (n_windows, 384)
        if agg == "mean_std":
            return torch.cat([embs.mean(0), embs.std(0, unbiased=False)]).numpy()
        return embs.mean(0).numpy()

    def embed_all(self, samples, agg, cache_name):
        """Extract (or load-cached) one embedding per sample -> (N, d) array."""
        cache = self.run_dir / cache_name
        if cache.exists() and int(np.load(cache)["X"].shape[0]) == len(samples):
            X = np.load(cache)["X"]
            print(f"loaded cached embeddings {X.shape} from {cache.name}", flush=True)
            return X
        t0, embs = time.time(), []
        for i, s in enumerate(samples):
            embs.append(self.cls_token(s["path"], s["tr"], agg=agg))
            if (i + 1) % 100 == 0 or i == len(samples) - 1:
                dt = time.time() - t0
                print(f"  embeddings {i+1}/{len(samples)}  ({dt:.0f}s, "
                      f"{dt/(i+1)*1000:.0f} ms/scan)", flush=True)
        X = np.stack(embs)
        np.savez(cache, X=X)
        print(f"cached embeddings -> {cache.name}", flush=True)
        return X


# =====================================================================
# 2. METRICS — one function per metric
# =====================================================================

def auroc(y_true, proba):
    return float(roc_auc_score(y_true, proba))


def accuracy(y_true, pred):
    return float(accuracy_score(y_true, pred))


def f1(y_true, pred):
    return float(f1_score(y_true, pred, zero_division=0))


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


# =====================================================================
# 3. EVALUATION — one function per task type
#    (each returns a plain dict written verbatim into the output JSON)
# =====================================================================

def _make_clf(head, hp):
    """Fresh classifier on the FROZEN embeddings. linear -> LogReg(C=hp);
    mlp -> MLPClassifier with hp=(hidden_layer_sizes, alpha)."""
    if head == "mlp":
        arch, alpha = hp
        return MLPClassifier(hidden_layer_sizes=arch, activation="relu", alpha=alpha,
                             max_iter=500, early_stopping=True, n_iter_no_change=15,
                             random_state=0)
    return LogisticRegression(C=hp, max_iter=2000, class_weight="balanced")


def _select_clf_hp(X, y, groups, head, mlp_archs=None):
    """Choose C (linear) / (arch, alpha) (mlp) by subject-aware CV on TRAIN,
    maximizing mean CV AUC. mlp_archs overrides the arch grid (per-arch ablation)."""
    archs = mlp_archs or MLP_ARCHS
    grid = ([(a, al) for a in archs for al in ALPHA_GRID] if head == "mlp" else C_GRID)
    default = ((256, 128), 1e-3) if head == "mlp" else 1.0
    yb = y.astype(int)
    grp_per_class = {c: len(set(groups[yb == c])) for c in np.unique(yb)}
    if min(grp_per_class.values()) < 2:
        return default
    n_splits = max(2, min(5, min(grp_per_class.values())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_hp, best = default, -1.0
    for hp in grid:
        aucs = []
        try:
            for tr, va in cv.split(X, yb, groups):
                if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[va])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = _make_clf(head, hp).fit(sc.transform(X[tr]), y[tr])
                aucs.append(auroc(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
        except ValueError:
            continue
        if aucs and np.mean(aucs) > best:
            best, best_hp = float(np.mean(aucs)), hp
    return best_hp


def _select_ridge_alpha(X, y, groups):
    """Ridge alpha by subject-aware GroupKFold CV on TRAIN, maximizing Pearson r."""
    if len(set(groups)) < 3:
        return 10.0
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    best_a, best = 10.0, -2.0
    for a in RIDGE_ALPHA_GRID:
        rs = []
        for tr, va in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            reg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
            r = pearson(reg.predict(sc.transform(X[va])), y[va])
            if not np.isnan(r):
                rs.append(r)
        if rs and np.mean(rs) > best:
            best, best_a = float(np.mean(rs)), a
    return best_a


def evaluate_binary_split(X, y, splits, groups, head="linear", mlp_archs=None):
    """70:30. TRAIN='train' subjects; TEST='val'+'test' (held out of pretraining).
    Head hyperparam by CV on TRAIN. Report TEST AUC/Acc/F1."""
    tr = (splits == "train")
    te = (splits == "val") | (splits == "test")
    m = ~np.isnan(y)
    tr, te = tr & m, te & m
    if tr.sum() < 10 or te.sum() < 10:
        return None
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    hp = _select_clf_hp(X[tr], y[tr], groups[tr], head, mlp_archs)
    sc = StandardScaler().fit(X[tr])
    clf = _make_clf(head, hp).fit(sc.transform(X[tr]), y[tr])
    proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"head": head, "hp": float(hp) if isinstance(hp, (int, float)) else hp,
            "test_auc": auroc(y[te], proba), "test_acc": accuracy(y[te], pred),
            "test_f1": f1(y[te], pred), "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_test_subj": int(len(set(groups[te]))), "pos_test": int(y[te].sum())}


def evaluate_binary_kfold(X, y, groups, n_splits=5, head="linear", mlp_archs=None):
    """Subject-aware k-fold over the WHOLE cohort (for datasets never pretrained on).
    Fixed hyperparam: linear C=1; mlp = given arch (or 256,128) with alpha=1e-3."""
    m = ~np.isnan(y)
    X, y, groups = X[m], y[m], groups[m]
    if len(np.unique(y)) < 2 or len(set(groups)) < n_splits:
        return None
    hp = ((mlp_archs[0] if mlp_archs else (256, 128)), 1e-3) if head == "mlp" else 1.0
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs, accs, f1s = [], [], []
    for tr, te in cv.split(X, y.astype(int), groups):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = _make_clf(head, hp).fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(auroc(y[te], proba)); accs.append(accuracy(y[te], pred)); f1s.append(f1(y[te], pred))
    if not aucs:
        return None
    return {"head": head, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "n": int(len(y)), "pos": int(y.sum()), "folds": len(aucs)}


def evaluate_multiclass_kfold(X, y, groups, n_splits=5):
    """Subject-aware k-fold multinomial LogReg for >2-class tasks (HCP task-state).
    Report mean±std accuracy and macro one-vs-rest AUC."""
    m = ~np.isnan(y)
    X, y, groups = X[m], y[m], groups[m]
    y = y.astype(int)
    classes = np.unique(y)
    if len(classes) < 3 or len(set(groups)) < n_splits:
        return None
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs, aucs = [], []
    for tr, te in cv.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        accs.append(accuracy(y[te], clf.predict(Xte)))
        try:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte), multi_class="ovr",
                                      average="macro", labels=clf.classes_))
        except ValueError:
            pass
    if not accs:
        return None
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_std": float(np.std(aucs)) if aucs else None,
            "n": int(len(y)), "n_classes": int(len(classes)), "folds": len(accs)}


def evaluate_regression_split(X, y, splits, groups, **_):
    """70:30 Ridge regression. alpha by CV on TRAIN. Report TEST Pearson r + R2 + MSE."""
    tr = (splits == "train")
    te = (splits == "val") | (splits == "test")
    m = ~np.isnan(y)
    tr, te = tr & m, te & m
    if tr.sum() < 20 or te.sum() < 10:
        return None
    a = _select_ridge_alpha(X[tr], y[tr], groups[tr])
    sc = StandardScaler().fit(X[tr])
    reg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
    pred = reg.predict(sc.transform(X[te]))
    yte = y[te]
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return {"alpha": float(a), "test_r": pearson(pred, yte),
            "test_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "test_mse": float(np.mean((pred - yte) ** 2)),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_test_subj": int(len(set(groups[te])))}


# =====================================================================
# 4. TASK + DATASET — one Dataset subclass per cohort
# =====================================================================

@dataclass
class Task:
    """One evaluation target: a label column, its kind, the metric reported vs the
    SOTA, and the SOTA reference string. `kind` in {binary, multiclass, regression}."""
    name: str
    column: str
    kind: str = "binary"
    metric: str = "test_auc"
    sota: str = ""


# ---- shared label helpers (used by several datasets) ----

def _split_map(ds):
    split = json.loads((LAB / DEFAULT_SPLIT).read_text())["datasets"][ds]
    return {s: name for name, subs in split.items() for s in subs}


def _add_age_bin(samples, age_field, key="age_bin"):
    ages = [_to_float(s["labels"].get(age_field)) for s in samples]
    valid = [a for a in ages if a is not None]
    med = float(np.median(valid)) if valid else None
    for s, a in zip(samples, ages):
        s["labels"][key] = (float("nan") if a is None or med is None
                            else (1.0 if a >= med else 0.0))


def _detect_col(header, wants):
    low = [h.strip().lower() for h in header]
    for want in wants:
        for i, h in enumerate(low):
            if want in h:
                return i
    return None


class Dataset(ABC):
    """Base cohort. Subclasses set `name`, `eval_mode` ('split' or 'kfold'), the
    `tasks` list, and implement `samples()` returning dicts with keys:
        path, subject, split ('train'/'val'/'test' or 'none'), tr, labels{col: value}."""
    name = ""
    eval_mode = "split"          # 'split' (70:30, in-pretraining) or 'kfold' (downstream-only)
    tasks: list = []
    cache_key = None             # override to reuse another dataset's embedding cache

    @abstractmethod
    def samples(self):
        ...


# ---------- ADNI ----------

class ADNI(Dataset):
    name = "ADNI"
    eval_mode = "split"
    DIR = LAB / "ADNI_data" / "downsampled"
    MANIFEST = DIR / "adni_manifest.csv"
    tasks = [
        Task("NC_vs_MCI", "nc_vs_mci", "binary", "test_acc", "Brain-JEPA 0.77 (FT)"),
        Task("AD_vs_HC", "ad_vs_hc", "binary", "test_auc", "BrainGFM 0.80"),
        Task("Amyloid", "amyloid_positive", "binary", "test_acc", "Brain-JEPA 0.71 (FT)"),
    ]

    def _clinical(self):
        """Real DX + amyloid (ADNIMERGE2 export), overrides the CDR proxy."""
        p = next((c for c in (self.DIR / "adni_clinical.csv",
                              REPO_ROOT / "data" / "adni_clinical.csv") if c.exists()), None)
        if p is None:
            return {}
        return {r["subject_id"]: {k: _to_float(r.get(k))
                for k in ("nc_vs_mci", "ad_vs_hc", "amyloid_positive")}
                for r in csv.DictReader(open(p))}

    def samples(self):
        sub2split = _split_map("ADNI")
        clinical = self._clinical()
        if clinical:
            print(f"ADNI: real clinical labels for {len(clinical)} subjects", flush=True)
        out = []
        for r in csv.DictReader(open(self.MANIFEST)):
            sid, iid = r["subject_id"], r["image_id"]
            p = self.DIR / sid / f"{iid}.pt"
            sp = sub2split.get(sid)
            if sp is None or not p.exists():
                continue
            cdr = _to_float(r.get("Global CDR"))     # proxy: NC=0, MCI=0.5, AD>=1
            lab = {"nc_vs_mci": (0.0 if cdr == 0 else 1.0 if cdr == 0.5 else float("nan")),
                   "ad_vs_hc": (0.0 if cdr == 0 else 1.0 if (cdr is not None and cdr >= 1) else float("nan")),
                   "amyloid_positive": float("nan"), "Age": r.get("Age")}
            c = clinical.get(sid)
            if c:
                for k in ("nc_vs_mci", "ad_vs_hc", "amyloid_positive"):
                    if c.get(k) is not None:
                        lab[k] = c[k]
            out.append({"path": p, "subject": sid, "split": sp, "tr": 3.0, "labels": lab})
        return out


# ---------- ABIDE ----------

class ABIDE(Dataset):
    name = "ABIDE"
    eval_mode = "split"
    PHENO = LAB / "ABIDE_data" / "abide_phenotypic.csv"
    tasks = [
        Task("Autism", "autism", "binary", "test_auc", "BNT 0.80 / BrainGFM 0.71"),
        Task("Age", "age_bin", "binary", "test_acc", "SLIM 0.64"),
        Task("Sex", "sex_bin", "binary", "test_f1", "LCM 0.87"),
    ]

    def samples(self):
        sub2split = _split_map("ABIDE")
        pheno = {r["FILE_ID"]: r for r in csv.DictReader(open(self.PHENO))}
        out = []
        for r in csv.DictReader(open(CORPUS_MANIFEST)):
            if r["dataset"] != "ABIDE":
                continue
            sid = r["subject_id"]
            sp = sub2split.get(sid)
            ph = pheno.get(sid.replace("_downsampled", ""))
            if sp is None or ph is None:
                continue
            dx = _to_float(ph.get("DX_GROUP"))       # 1=autism, 2=control
            if dx is None:
                continue
            sex = _to_float(ph.get("SEX"))           # 1=male, 2=female
            lab = {"autism": 1.0 if dx == 1 else 0.0,
                   "sex_bin": (1.0 if sex == 1 else 0.0) if sex is not None else float("nan"),
                   "AGE_AT_SCAN": ph.get("AGE_AT_SCAN")}
            out.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                        "tr": float(r["tr"]), "labels": lab})
        _add_age_bin(out, "AGE_AT_SCAN")
        return out


# ---------- HCP (rest: Sex / Age) ----------

class HCP(Dataset):
    name = "HCP"
    eval_mode = "split"
    CSV = REPO_ROOT / "data" / "HCP_YA_subjects.csv"
    tasks = [
        Task("Sex", "sex_bin", "binary", "test_acc", "SLIM 0.91"),
        Task("Age", "age_bin", "binary", "test_acc", "—"),
    ]

    def _meta(self):
        return {str(r["Subject"]).strip(): r for r in csv.DictReader(open(self.CSV))}

    def samples(self):
        sub2split = _split_map("HCP")
        meta = self._meta()
        out = []
        for r in csv.DictReader(open(CORPUS_MANIFEST)):
            if r["dataset"] != "HCP":
                continue
            sid = r["subject_id"]
            sp = sub2split.get(sid)
            if sp is None:
                continue
            digits = "".join(ch for ch in sid if ch.isdigit())[:6]
            m = meta.get(sid) or meta.get(digits)
            if m is None:
                continue
            g = (m.get("Gender") or "").strip()
            lab = {"sex_bin": 1.0 if g == "M" else 0.0 if g == "F" else float("nan"),
                   "Age_in_Yrs": m.get("Age_in_Yrs")}
            out.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                        "tr": float(r["tr"]), "labels": lab})
        _add_age_bin(out, "Age_in_Yrs")
        print(f"HCP: matched {len(out)} scans", flush=True)
        return out


# ---------- HCP cognition (regression, reuses HCP scans/cache) ----------

class HCP_COG(HCP):
    name = "HCP_COG"
    eval_mode = "split"
    cache_key = "HCP"                 # identical scans/order -> reuse HCP embeddings
    TARGETS = {"FluidIntel": "PMAT24_A_CR", "ProcSpeed": "ProcSpeed_AgeAdj",
               "WorkingMem": "ListSort_AgeAdj"}
    tasks = [
        Task("FluidIntel", "fluidintel", "regression", "test_r", "—"),
        Task("ProcSpeed", "procspeed", "regression", "test_r", "—"),
        Task("WorkingMem", "workingmem", "regression", "test_r", "—"),
    ]

    def samples(self):
        sub2split = _split_map("HCP")
        meta = self._meta()
        out = []
        for r in csv.DictReader(open(CORPUS_MANIFEST)):
            if r["dataset"] != "HCP":
                continue
            sid = r["subject_id"]
            sp = sub2split.get(sid)
            if sp is None:
                continue
            digits = "".join(ch for ch in sid if ch.isdigit())[:6]
            m = meta.get(sid) or meta.get(digits)
            if m is None:
                continue
            lab = {t.column: _to_float(m.get(self.TARGETS[t.name])) for t in self.tasks}
            out.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                        "tr": float(r["tr"]), "labels": lab})
        print(f"HCP_COG: matched {len(out)} scans", flush=True)
        return out


# ---------- OASIS (AD Conversion) ----------

class OASIS(Dataset):
    name = "OASIS"
    eval_mode = "split"
    LABELS = LAB / "OASIS3_data" / "oasis_labels.csv"
    tasks = [Task("AD_Conversion", "ad_conversion", "binary", "test_acc", "—")]

    def samples(self):
        sub2split = _split_map("OASIS")
        labels = {}
        if self.LABELS.exists():
            for r in csv.DictReader(open(self.LABELS)):
                labels[r["subject_id"]] = _to_float(r.get("ad_conversion"))
        else:
            print(f"OASIS: {self.LABELS.name} not found -> run derive_oasis_adconv.py", flush=True)
        out = []
        for r in csv.DictReader(open(CORPUS_MANIFEST)):
            if r["dataset"] != "OASIS":
                continue
            sid = r["subject_id"]
            sp = sub2split.get(sid)
            if sp is None:
                continue
            key = next((k for k in (sid, sid.split("_")[0]) if k in labels), None)
            out.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                        "tr": float(r["tr"]), "labels": {"ad_conversion": labels.get(key, float("nan"))}})
        return out


# ---------- downstream-only cohorts (k-fold, never pretrained on) ----------

class ADHD(Dataset):
    name = "ADHD"
    eval_mode = "kfold"
    DIR = LAB / "ADHD200_data" / "downsampled"
    PHENO = LAB / "ADHD200_data" / "adhd200_phenotypic.csv"
    SITE_TR = {"Peking": 2.0, "KKI": 2.5, "NYU": 2.0, "NeuroIMAGE": 1.96,
               "OHSU": 2.5, "Pittsburgh": 1.5, "Brown": 2.5, "WashU": 2.5}
    tasks = [Task("ADHD", "adhd", "binary", "test_acc", "NeuroSTORM 0.587 (ADHD-200)")]

    def samples(self):
        if not self.PHENO.exists():
            print(f"ADHD: {self.PHENO} not found", flush=True)
            return []
        rows = list(csv.reader(open(self.PHENO)))
        header = rows[0]
        si = _detect_col(header, ["scandir", "subject"]) or 0
        di = _detect_col(header, ["dx"])
        gi = _detect_col(header, ["site"])
        pheno = {}
        for r in rows[1:]:
            if di is None or si >= len(r) or di >= len(r):
                continue
            digits = "".join(ch for ch in r[si] if ch.isdigit())
            dxv = _to_float(r[di])
            if not digits or dxv is None:
                continue
            site = r[gi].strip() if gi is not None and gi < len(r) else ""
            pheno[int(digits)] = (0.0 if dxv == 0 else 1.0, site)
        out = []
        for p in sorted(self.DIR.glob("*_downsampled.pt")):
            stem = p.name.replace("_downsampled.pt", "")
            digits = "".join(ch for ch in stem.split("_session")[0] if ch.isdigit())
            rec = pheno.get(int(digits)) if digits else None
            if rec is None:
                continue
            adhd, site = rec
            tr = next((v for k, v in self.SITE_TR.items() if k.lower() in site.lower()), 2.0)
            out.append({"path": p, "subject": stem.split("_session")[0], "split": "none",
                        "tr": tr, "labels": {"adhd": adhd}})
        print(f"ADHD: {len(out)} scans matched", flush=True)
        return out


class COBRE(Dataset):
    name = "COBRE"
    eval_mode = "kfold"
    DIR = LAB / "COBRE_data" / "downsampled"
    LABELS = LAB / "COBRE_data" / "cobre_labels.csv"
    tasks = [Task("Schizophrenia", "sz", "binary", "test_acc", "—")]

    def samples(self):
        labels = {}
        if self.LABELS.exists():
            for r in csv.DictReader(open(self.LABELS)):
                labels[r["subject_id"].strip()] = _to_float(r.get("sz"))
        else:
            print(f"COBRE: {self.LABELS} not found -> run download_cobre.py", flush=True)
        out = []
        for p in sorted(self.DIR.glob("*_downsampled.pt")):
            subj = p.name.replace("_downsampled.pt", "")
            sz = labels.get(subj)
            if sz is None:
                continue
            out.append({"path": p, "subject": subj, "split": "none", "tr": 2.0,
                        "labels": {"sz": sz}})
        print(f"COBRE: {len(out)} scans matched", flush=True)
        return out


class UCLA(Dataset):
    name = "UCLA"
    eval_mode = "kfold"
    DIR = LAB / "UCLA_data" / "downsampled"
    LABELS = LAB / "UCLA_data" / "ucla_participants.tsv"
    tasks = [
        Task("Schizophrenia", "schizophrenia", "binary", "test_acc", "—"),
        Task("ADHD", "adhd", "binary", "test_auc", "NeuroSTORM 0.604 (UCLA)"),
    ]

    def samples(self):
        dx = {}
        if self.LABELS.exists():
            for r in csv.DictReader(open(self.LABELS), delimiter="\t"):
                dx[r["participant_id"].strip()] = (r.get("diagnosis") or "").strip().upper()
        else:
            print(f"UCLA: {self.LABELS} not found -> run download_ucla.py", flush=True)
        out = []
        for p in sorted(self.DIR.glob("*_downsampled.pt")):
            subj = p.name.replace("_downsampled.pt", "")
            d = dx.get(subj)
            if d is None:
                continue
            lab = {"schizophrenia": (1.0 if d == "SCHZ" else 0.0 if d == "CONTROL" else float("nan")),
                   "adhd": (1.0 if d == "ADHD" else 0.0 if d == "CONTROL" else float("nan"))}
            out.append({"path": p, "subject": subj, "split": "none", "tr": 2.0, "labels": lab})
        print(f"UCLA: {len(out)} scans matched", flush=True)
        return out


class HCP_TASK(Dataset):
    name = "HCP_TASK"
    eval_mode = "kfold"
    MANIFEST = LAB / "HCP_task_data" / "hcp_task_labels.csv"
    TASKS_7 = ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
    tasks = [Task("TaskState", "task_id", "multiclass", "acc_mean", "—")]

    def samples(self):
        tid = {t: i for i, t in enumerate(self.TASKS_7)}
        if not self.MANIFEST.exists():
            print(f"HCP_TASK: {self.MANIFEST} not found -> run download_hcp_task.py", flush=True)
            return []
        out = []
        for r in csv.DictReader(open(self.MANIFEST)):
            p = Path(r["path"])
            t = tid.get(r["task"])
            if t is None or not p.exists():
                continue
            out.append({"path": p, "subject": r["subject"], "split": "none",
                        "tr": 0.72, "labels": {"task_id": float(t)}})
        print(f"HCP_TASK: {len(out)} runs", flush=True)
        return out


REGISTRY = {c.name: c for c in
            [ADNI, ABIDE, HCP, HCP_COG, OASIS, ADHD, COBRE, UCLA, HCP_TASK]}


# =====================================================================
# RUNNER — samples -> embeddings -> evaluate each task -> JSON
# =====================================================================

def evaluate_task(task, X, y, splits, groups, eval_mode, head, mlp_archs, n_splits):
    """Dispatch to the protocol matching (task kind × dataset eval mode)."""
    if task.kind == "regression":
        return evaluate_regression_split(X, y, splits, groups)
    if task.kind == "multiclass":
        return evaluate_multiclass_kfold(X, y, groups, n_splits=n_splits)
    if eval_mode == "kfold":
        return evaluate_binary_kfold(X, y, groups, n_splits, head, mlp_archs)
    return evaluate_binary_split(X, y, splits, groups, head, mlp_archs)


def run_probe(dataset, encoder, head="linear", agg="mean", n_splits=5, out=None):
    samples = dataset.samples()
    print(f"{dataset.name}: {len(samples)} scans with split+label", flush=True)
    if not samples:
        raise RuntimeError(f"No {dataset.name} scans matched (labels/split missing?)")

    ck = encoder.checkpoint.replace(".", "_")
    agg_suffix = "" if agg == "mean" else f"_{agg}"
    cache_name = f"emb_{dataset.cache_key or dataset.name}_{ck}{agg_suffix}.npz"
    X = encoder.embed_all(samples, agg, cache_name)
    splits = np.array([s["split"] for s in samples])
    groups = np.array([s["subject"] for s in samples])

    mode = f"{n_splits}-fold CV" if dataset.eval_mode == "kfold" else "70:30 split"
    print(f"\n{'='*66}\n  PROBE  {encoder.run_dir.name}  {dataset.name}  "
          f"[{mode}, head={head}, agg={agg}]\n{'='*66}", flush=True)
    results = {}
    for task in dataset.tasks:
        y = np.array([_to_float(s["labels"].get(task.column)) for s in samples], dtype=float)
        r = evaluate_task(task, X, y, splits, groups, dataset.eval_mode, head,
                          None, n_splits)
        results[task.name] = r
        head_metric = (r.get(task.metric) if r else None)
        hm = f"{head_metric:.3f}" if isinstance(head_metric, float) else "—"
        print(f"  {task.name:16} {task.metric:9} = {hm:>6}   (SOTA: {task.sota})", flush=True)

    payload = {"run": encoder.run_dir.name, "dataset": dataset.name, "mode": mode,
               "head": head, "agg": agg, "checkpoint": encoder.checkpoint, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nSaved {out}", flush=True)
    return payload


def run_probe_with_arch(dataset, encoder, head, agg, n_splits, out, mlp_archs):
    """Same as run_probe but threads a forced MLP arch into the evaluation (for the
    per-arch ablation). Kept separate so run_probe stays simple."""
    samples = dataset.samples()
    print(f"{dataset.name}: {len(samples)} scans", flush=True)
    if not samples:
        raise RuntimeError(f"No {dataset.name} scans matched")
    ck = encoder.checkpoint.replace(".", "_")
    agg_suffix = "" if agg == "mean" else f"_{agg}"
    cache_name = f"emb_{dataset.cache_key or dataset.name}_{ck}{agg_suffix}.npz"
    X = encoder.embed_all(samples, agg, cache_name)
    splits = np.array([s["split"] for s in samples])
    groups = np.array([s["subject"] for s in samples])
    mode = f"{n_splits}-fold CV" if dataset.eval_mode == "kfold" else "70:30 split"
    print(f"\n  PROBE {encoder.run_dir.name} {dataset.name} [{mode}, head={head}, "
          f"agg={agg}, mlp_arch={mlp_archs}]", flush=True)
    results = {}
    for task in dataset.tasks:
        y = np.array([_to_float(s["labels"].get(task.column)) for s in samples], dtype=float)
        results[task.name] = evaluate_task(task, X, y, splits, groups, dataset.eval_mode,
                                            head, mlp_archs, n_splits)
    payload = {"run": encoder.run_dir.name, "dataset": dataset.name, "mode": mode,
               "head": head, "agg": agg, "checkpoint": encoder.checkpoint, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"Saved {out}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="ADNI", choices=sorted(REGISTRY))
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--agg", default="mean", choices=["mean", "mean_std"])
    ap.add_argument("--mlp-arch", default=None, help="force one MLP arch, e.g. '256,128'")
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    ap.add_argument("--kfold", type=int, default=5, help="n folds for kfold datasets")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Device: {DEVICE}  cuda={torch.cuda.is_available()}", flush=True)
    encoder = Encoder(args.run_dir, args.checkpoint)
    dataset = REGISTRY[args.dataset]()
    mlp_archs = ([tuple(int(w) for w in args.mlp_arch.split(","))] if args.mlp_arch else None)

    if mlp_archs:
        run_probe_with_arch(dataset, encoder, args.head, args.agg, args.kfold,
                            args.out, mlp_archs)
    else:
        run_probe(dataset, encoder, args.head, args.agg, args.kfold, args.out)


if __name__ == "__main__":
    main()
