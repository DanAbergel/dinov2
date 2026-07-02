"""Leakage-free linear probe (ADNI / ABIDE / HCP) for a trained fMRI run.

Loads the run's teacher encoder, extracts ONE embedding per scan (mean CLS over
sliding windows resampled from the scan's native TR to 0.72s), then for each
principal SOTA axis:
  - TRAIN = 70% subjects, TEST = the held-out 30% (val+test merged).
  - the LogReg regularisation C is chosen by SUBJECT-AWARE cross-validation on the
    TRAIN portion only (no separate val set needed; the 30% test stays untouched).
  - report TEST AUC / Accuracy / F1 (binary, positive class — matches SOTA).
Subjects come from subject_split.json; the 30% test was held out of pretraining
(HOLDOUT_DATASETS), so the encoder never saw it -> no leakage.

Usage (SLURM job, needs GPU):
    python probe.py --run-dir .../runs/v2/base --dataset ADNI
    python probe.py --run-dir .../runs/v2/base --dataset ABIDE
    python probe.py --run-dir .../runs/v2/base --dataset HCP
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from dinov2.models import build_model_from_cfg
from dinov2.data.fmri_data import (
    _load_mmap, _temporal_resample, _zscore_per_frame,
    TARGET_TR, LAB_ROOT, DEFAULT_SPLIT, DEFAULT_MANIFEST,
)

LAB = Path(LAB_ROOT)
REPO_ROOT = Path(__file__).resolve().parents[3]     # tasks/v2/probe/probe.py -> repo root
ADNI_DIR = LAB / "ADNI_data" / "downsampled"
ADNI_MANIFEST = ADNI_DIR / "adni_manifest.csv"
ABIDE_PHENO = LAB / "ABIDE_data" / "abide_phenotypic.csv"
HCP_CSV = REPO_ROOT / "data" / "HCP_YA_subjects.csv"  # Subject,Gender,Age_in_Yrs,...
CORPUS_MANIFEST = LAB / DEFAULT_MANIFEST
T_FIXED = 270
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C_GRID = [0.01, 0.1, 1.0, 10.0]              # linear (LogReg) inverse-reg
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]        # MLP L2 regularisation
MLP_HIDDEN = (256, 128)                       # 2 hidden layers ("MLP, multiple layers")

# Principal SOTA axes only (degradation/CDR are novel-no-SOTA -> dropped from the
# headline; the columns stay computed in the table so they can be re-added later).
LABELS_ADNI = {
    # Diagnostic classification, proxy from Global CDR (Sagi's manifest has no
    # clinical DX): NC=CDR 0, MCI=CDR 0.5, AD=CDR>=1.
    "NC_vs_MCI": "nc_vs_mci",          # vs Brain-JEPA 0.77 / BNT 0.79 acc
    "AD_vs_HC": "ad_vs_hc",            # vs BrainGFM 0.80 AUC / LCM 0.85 F1
    "Amyloid": "amyloid_positive",     # Brain-JEPA task; needs adni_clinical.csv (ADNIMERGE join)
}
LABELS_ABIDE = {"Autism": "autism", "Age": "age_bin", "Sex": "sex_bin"}
LABELS_HCP = {"Sex": "sex_bin", "Age": "age_bin"}   # Sex = SOTA axis (SLIM 0.91 / LCM 0.73 F1)
LABELS_OASIS = {"AD_Conversion": "ad_conversion"}   # Brain-JEPA task (0.69 acc); needs oasis_labels.csv
OASIS_LABELS = LAB / "OASIS3_data" / "oasis_labels.csv"


# ---------------- encoder ----------------

def load_teacher(run_dir, checkpoint):
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    _student, teacher, embed_dim = build_model_from_cfg(cfg)
    ckpt = torch.load(Path(run_dir) / checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    tb = {}
    for k, v in state.items():
        kk = k.replace("_fsdp_wrapped_module.", "")
        if kk.startswith("teacher.backbone."):
            tb[kk[len("teacher.backbone."):]] = v
    if not tb:
        raise RuntimeError(f"No teacher.backbone.* keys. Sample: {list(state)[:8]}")
    missing, unexpected = teacher.load_state_dict(tb, strict=False)
    print(f"teacher: loaded {len(tb)} keys  missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    return teacher.to(DEVICE).eval(), embed_dim


@torch.no_grad()
def scan_embedding(teacher, path, native_tr):
    """Mean CLS over sliding native windows (each spanning T_FIXED*0.72s),
    resampled to T_FIXED frames at 0.72s. Cropping native FIRST keeps
    resample_poly cheap (small up/down)."""
    win = max(1, round(T_FIXED * TARGET_TR / native_tr))
    stride = max(1, win // 2)
    scan = _load_mmap(path).float()                  # (T, X, Y, Z)
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                     # (T, 1, X, Y, Z)
    T = scan.shape[0]
    embs = []
    for s in range(0, max(T - win + 1, 1), stride):
        clip = scan[s:s + win].clone()
        clip = _temporal_resample(clip, T_FIXED)     # -> 270 @ 0.72s
        clip = _zscore_per_frame(clip)
        x = clip.unsqueeze(0).to(DEVICE)             # (1, T_FIXED, 1, X, Y, Z)
        out = teacher(x, is_training=True)
        embs.append(out["x_norm_clstoken"].squeeze(0).float().cpu())
    return torch.stack(embs).mean(0).numpy()


# ---------------- labels / split ----------------

def _to_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _split_map(ds):
    split = json.loads((LAB / DEFAULT_SPLIT).read_text())["datasets"][ds]
    return {s: name for name, subs in split.items() for s in subs}


def _add_age_bin(table, age_field):
    """Add row['age_bin'] = 1 if age >= global median else 0 (NaN if missing).
    Global median (one scalar) works in both fixed-split and k-fold modes."""
    ages = [_to_float(t["row"].get(age_field)) for t in table]
    valid = [a for a in ages if a is not None]
    med = float(np.median(valid)) if valid else None
    for t, a in zip(table, ages):
        t["row"]["age_bin"] = (float("nan") if a is None or med is None
                               else (1.0 if a >= med else 0.0))


def _load_adni_clinical():
    """Optional real DX + amyloid from ADNIMERGE join (add_adni_labels.py).
    subject_id -> {nc_vs_mci, ad_vs_hc, amyloid_positive} (real, overrides CDR proxy)."""
    p = ADNI_DIR / "adni_clinical.csv"
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["subject_id"]] = {k: _to_float(r.get(k))
                                for k in ("nc_vs_mci", "ad_vs_hc", "amyloid_positive")}
    return out


def build_table_adni():
    sub2split = _split_map("ADNI")
    clinical = _load_adni_clinical()      # real DX + amyloid if available
    if clinical:
        print(f"ADNI: real clinical labels for {len(clinical)} subjects "
              f"(overriding CDR proxy where present)", flush=True)
    table = []
    for r in csv.DictReader(open(ADNI_MANIFEST)):
        sid, iid = r["subject_id"], r["image_id"]
        p = ADNI_DIR / sid / f"{iid}.pt"
        sp = sub2split.get(sid)
        if sp is None or not p.exists():
            continue
        # Derived diagnostic labels from Global CDR (NC=0, MCI=0.5, AD>=1).
        cdr = _to_float(r.get("Global CDR"))
        r["nc_vs_mci"] = (0.0 if cdr == 0 else 1.0 if cdr == 0.5 else float("nan"))
        r["ad_vs_hc"] = (0.0 if cdr == 0 else
                         1.0 if (cdr is not None and cdr >= 1) else float("nan"))
        r["amyloid_positive"] = float("nan")
        # override with REAL clinical labels + amyloid when the join file exists
        c = clinical.get(sid)
        if c:
            for k in ("nc_vs_mci", "ad_vs_hc", "amyloid_positive"):
                if c.get(k) is not None:
                    r[k] = c[k]
        table.append({"path": p, "subject": sid, "split": sp, "tr": 3.0, "row": r})
    _add_age_bin(table, "Age")         # binary age at global median
    return table, LABELS_ADNI


def build_table_abide():
    sub2split = _split_map("ABIDE")
    pheno = {r["FILE_ID"]: r for r in csv.DictReader(open(ABIDE_PHENO))}
    table = []
    for r in csv.DictReader(open(CORPUS_MANIFEST)):
        if r["dataset"] != "ABIDE":
            continue
        sid = r["subject_id"]
        sp = sub2split.get(sid)
        if sp is None:
            continue
        ph = pheno.get(sid.replace("_downsampled", ""))   # FILE_ID = name w/o suffix
        if ph is None:
            continue
        dx = _to_float(ph.get("DX_GROUP"))                # 1=autism, 2=control
        if dx is None:
            continue
        sex = _to_float(ph.get("SEX"))                    # 1=male, 2=female
        row = {"autism": 1.0 if dx == 1 else 0.0,
               "sex_bin": (1.0 if sex == 1 else 0.0) if sex is not None else float("nan"),
               "AGE_AT_SCAN": ph.get("AGE_AT_SCAN")}
        table.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                      "tr": float(r["tr"]), "row": row})
    _add_age_bin(table, "AGE_AT_SCAN")     # binary age at global median
    return table, LABELS_ABIDE


def build_table_hcp():
    """HCP-YA: Sex (Gender M/F) is the SOTA axis (SLIM-Brain 0.91, LCM 0.73 F1);
    Age (Age_in_Yrs) binarised at the median as a demographic sanity check."""
    sub2split = _split_map("HCP")
    meta = {str(r["Subject"]).strip(): r for r in csv.DictReader(open(HCP_CSV))}
    table, matched = [], 0
    for r in csv.DictReader(open(CORPUS_MANIFEST)):
        if r["dataset"] != "HCP":
            continue
        sid = r["subject_id"]
        sp = sub2split.get(sid)
        if sp is None:
            continue
        # HCP CSV "Subject" is a 6-digit id; manifest subject_id may carry a suffix.
        digits = "".join(ch for ch in sid if ch.isdigit())[:6]
        m = meta.get(sid) or meta.get(digits)
        if m is None:
            continue
        g = (m.get("Gender") or "").strip()
        row = {"sex_bin": 1.0 if g == "M" else 0.0 if g == "F" else float("nan"),
               "Age_in_Yrs": m.get("Age_in_Yrs")}
        table.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                      "tr": float(r["tr"]), "row": row})
        matched += 1
    print(f"HCP: matched {matched} scans to Sex/Age labels", flush=True)
    _add_age_bin(table, "Age_in_Yrs")
    return table, LABELS_HCP


def build_table_oasis():
    """OASIS-3 AD Conversion (Brain-JEPA task). Needs oasis_labels.csv from
    derive_oasis_adconv.py (CDR trajectory). Matches by the OAS3xxxx id."""
    sub2split = _split_map("OASIS")
    labels = {}
    if OASIS_LABELS.exists():
        for r in csv.DictReader(open(OASIS_LABELS)):
            labels[r["subject_id"]] = _to_float(r.get("ad_conversion"))
    else:
        print(f"OASIS: {OASIS_LABELS.name} not found -> run derive_oasis_adconv.py "
              f"(AD Conversion will be all-NaN)", flush=True)
    table, matched = [], 0
    for r in csv.DictReader(open(CORPUS_MANIFEST)):
        if r["dataset"] != "OASIS":
            continue
        sid = r["subject_id"]
        sp = sub2split.get(sid)
        if sp is None:
            continue
        key = next((k for k in (sid, sid.split("_")[0]) if k in labels), None)
        row = {"ad_conversion": labels.get(key, float("nan"))}
        if key:
            matched += 1
        table.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                      "tr": float(r["tr"]), "row": row})
    print(f"OASIS: {matched} scans matched to AD-Conversion labels", flush=True)
    return table, LABELS_OASIS


# ---------------- probe ----------------

def _make_clf(head, hp):
    """A fresh classifier on the FROZEN embeddings.
    head='linear' -> LogReg (C=hp, class-balanced); head='mlp' -> 2-hidden-layer
    MLP (alpha=hp). This is the point-3 ablation axis: linear probe vs MLP head,
    both on a frozen encoder (the encoder is NOT fine-tuned — that is point 4)."""
    if head == "mlp":
        return MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation="relu",
                             alpha=hp, max_iter=500, early_stopping=True,
                             n_iter_no_change=15, random_state=0)
    return LogisticRegression(C=hp, max_iter=2000, class_weight="balanced")


def _select_hp_cv(X, y, groups, head):
    """Choose the head's regularisation (C for linear, alpha for MLP) by
    SUBJECT-AWARE cross-validation on the TRAIN set only. The CV builds temporary
    validation folds from the train subjects (grouped), so the 30% test is never
    touched. Selection criterion = mean CV AUC. Falls back to a default if the
    train set is too small to split."""
    grid = ALPHA_GRID if head == "mlp" else C_GRID
    default = 1e-3 if head == "mlp" else 1.0
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
                aucs.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
        except ValueError:
            continue
        if aucs and np.mean(aucs) > best:
            best, best_hp = float(np.mean(aucs)), hp
    return best_hp


def probe_label(X, y, splits, groups, name, head="linear"):
    """70:30 split. TRAIN = 'train' subjects (in pretraining is fine); TEST = the
    held-out 30% ('val'+'test', excluded from pretraining). Head hyperparam chosen
    by CV on TRAIN. Report TEST AUC/Acc/F1 (F1 binary = positive class, matches SOTA)."""
    tr = (splits == "train")
    te = (splits == "val") | (splits == "test")     # 70:30 — merge val+test as held-out test
    m = ~np.isnan(y)
    tr, te = tr & m, te & m
    if tr.sum() < 10 or te.sum() < 10:
        return None
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None

    best_hp = _select_hp_cv(X[tr], y[tr], groups[tr], head)
    sc = StandardScaler().fit(X[tr])
    clf = _make_clf(head, best_hp).fit(sc.transform(X[tr]), y[tr])
    proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"head": head, "hp": float(best_hp),
            "test_auc": float(roc_auc_score(y[te], proba)),
            "test_acc": float(accuracy_score(y[te], pred)),
            "test_f1": float(f1_score(y[te], pred, zero_division=0)),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_test_subj": int(len(set(groups[te]))), "pos_test": int(y[te].sum())}


def probe_kfold(X, y, groups, n_splits=5):
    """Subject-aware k-fold CV: every subject is a test sample once. Use this when
    the encoder saw NONE of these subjects in pretraining (e.g. ADNI fully excluded)
    -> stable estimate over the full cohort, comparable to SOTA test sizes. Fixed
    C=1 (no val tuning), report mean +/- std of AUC and Accuracy across folds."""
    m = ~np.isnan(y)
    X, y, groups = X[m], y[m], groups[m]
    if len(np.unique(y)) < 2 or len(set(groups)) < n_splits:
        return None
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs, accs, f1s = [], [], []
    for tr, te in cv.split(X, y.astype(int), groups):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, zero_division=0))
    if not aucs:
        return None
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "n": int(len(y)), "pos": int(y.sum()), "folds": len(aucs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="ADNI", choices=["ADNI", "ABIDE", "HCP", "OASIS"])
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"],
                    help="probe head on the frozen encoder (point-3 ablation): "
                         "linear=LogReg, mlp=2-hidden-layer MLP.")
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    ap.add_argument("--kfold", type=int, default=0,
                    help="If >0, subject-aware k-fold CV over the whole cohort "
                         "(use only when this dataset was FULLY excluded from "
                         "pretraining). Else fixed train/val/test split.")
    args = ap.parse_args()

    print(f"Device: {DEVICE}  cuda_available={torch.cuda.is_available()}", flush=True)
    teacher, embed_dim = load_teacher(args.run_dir, args.checkpoint)
    print(f"embed_dim={embed_dim}", flush=True)

    table, LABELS = ({"ADNI": build_table_adni, "ABIDE": build_table_abide,
                      "HCP": build_table_hcp, "OASIS": build_table_oasis}[args.dataset])()
    print(f"{args.dataset} scans with split+label: {len(table)}", flush=True)
    if not table:
        raise RuntimeError(f"No {args.dataset} scans matched (labels/split missing?)")

    # Cache embeddings per (dataset, checkpoint): extraction is the slow step, so
    # re-running for a new metric (F1, ...) is then instant.
    ck = args.checkpoint.replace(".", "_")
    cache = Path(args.run_dir) / f"emb_{args.dataset}_{ck}.npz"
    if cache.exists() and int(np.load(cache)["X"].shape[0]) == len(table):
        X = np.load(cache)["X"]
        print(f"loaded cached embeddings {X.shape} from {cache.name}", flush=True)
    else:
        t0 = time.time()
        embs = []
        for i, t in enumerate(table):
            embs.append(scan_embedding(teacher, t["path"], t["tr"]))
            if (i + 1) % 100 == 0 or i == len(table) - 1:
                dt = time.time() - t0
                print(f"  embeddings {i+1}/{len(table)}  ({dt:.0f}s, "
                      f"{dt/(i+1)*1000:.0f} ms/scan)", flush=True)
        X = np.stack(embs)
        np.savez(cache, X=X)
        print(f"cached embeddings -> {cache.name}", flush=True)
    splits = np.array([t["split"] for t in table])

    run_name = Path(args.run_dir).name
    mode = f"{args.kfold}-fold CV (full cohort)" if args.kfold else "fixed split (val-select)"
    print(f"\n{'='*66}\n  PROBE  {run_name}  {args.dataset}  [{mode}]\n{'='*66}")
    results = {}

    if args.kfold:
        groups = np.array([t["subject"] for t in table])
        print(f"  {'label':16} {'AUC (mean±std)':>18} {'Acc (mean±std)':>18}  {'n':>5} pos")
        for name, col in LABELS.items():
            y = np.array([_to_float(t["row"].get(col)) for t in table], dtype=float)
            r = probe_kfold(X, y, groups, n_splits=args.kfold)
            results[name] = r
            if r:
                print(f"  {name:16} AUC {r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
                      f"Acc {r['acc_mean']:.3f}±{r['acc_std']:.3f}  "
                      f"F1 {r['f1_mean']:.3f}±{r['f1_std']:.3f}  n={r['n']} pos={r['pos']}",
                      flush=True)
            else:
                print(f"  {name:16} (skipped)", flush=True)
    else:
        groups = np.array([t["subject"] for t in table])
        print(f"  head={args.head}")
        print(f"  {'label':16} {'hp':>8} {'TEST_auc':>9} {'acc':>6} {'F1':>6}  "
              f"{'n_te':>5} {'subj':>5} pos")
        for name, col in LABELS.items():
            y = np.array([_to_float(t["row"].get(col)) for t in table], dtype=float)
            r = probe_label(X, y, splits, groups, name, head=args.head)
            results[name] = r
            if r:
                print(f"  {name:16} {r['hp']:>8.4g} {r['test_auc']:>9.2f}"
                      f" {r['test_acc']:>6.2f} {r['test_f1']:>6.2f}  {r['n_test']:>5} "
                      f"{r['n_test_subj']:>5} {r['pos_test']}", flush=True)
            else:
                print(f"  {name:16} (skipped — too few samples / one class)", flush=True)

    # linear -> probe_<ds>.json (default); mlp -> probe_<ds>_mlp.json (point-3 ablation)
    head_suffix = "" if args.head == "linear" else f"_{args.head}"
    suffix = f"_kfold{args.kfold}" if args.kfold else ""
    out = Path(args.run_dir) / f"probe_{args.dataset.lower()}{head_suffix}{suffix}.json"
    out.write_text(json.dumps({"run": run_name, "dataset": args.dataset, "mode": mode,
                               "head": args.head, "checkpoint": args.checkpoint,
                               "results": results}, indent=2))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
