"""Per-label baselines (chance-level) for HCP and ADNI probe metrics.

Computes the minimum-effort references for each downstream label:
  - Classification: sklearn DummyClassifier(strategy='stratified') AUC,
                    and the prevalence (= AUPRC baseline).
  - Regression:     sklearn DummyRegressor(strategy='mean') MAE.

These are the numbers that appear in the "Chance / Mean baseline" column
of the FREEZE_ABLATION_RESULTS doc -- the floor that any non-trivial probe
must beat to claim signal.

Run on Moriah from FAIR_official root:
    python3 scripts/compute_label_baselines.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

FAIR_REPO = Path(os.environ.get(
    "FAIR_DIR", "/sci/labs/arieljaffe/dan.abergel1/repos/FAIR"))
sys.path.insert(0, str(FAIR_REPO))

# ---- HCP labels ----
from src.config import HCP_LABELS, HCP_ROOT, HCP_SUBJECTS_CSV, N_SPLITS, RANDOM_STATE
from src.baselines.utils import get_label_array as get_hcp_label
import pandas as pd
import re

# ---- ADNI labels ----
from src.config import ADNI_INDEX_JSON, ADNI_LABELS, ADNI_LABELS_JSON
from src.baselines.utils import load_adni_labels, get_label_array as get_adni_label


def dummy_clf_auc(y, n_splits=5):
    """Stratified k-fold AUC of a stratified-random DummyClassifier."""
    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        return None
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, val in cv.split(np.zeros((len(y), 1)), y):
        clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
        clf.fit(np.zeros((len(tr), 1)), y[tr])
        proba = clf.predict_proba(np.zeros((len(val), 1)))[:, 1]
        aucs.append(roc_auc_score(y[val], proba))
    return float(np.mean(aucs))


def dummy_reg_mae(y, n_splits=5):
    """k-fold MAE of DummyRegressor(strategy='mean')."""
    y = y.astype(np.float64)
    cv = KFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes = []
    for tr, val in cv.split(np.zeros((len(y), 1))):
        reg = DummyRegressor(strategy="mean").fit(np.zeros((len(tr), 1)), y[tr])
        maes.append(mean_absolute_error(y[val], reg.predict(np.zeros((len(val), 1)))))
    return float(np.mean(maes))


# ---------- HCP ----------
print("=" * 65)
print("HCP labels -- chance baselines")
print("=" * 65)
print(f"{'Label':<16} {'n':>6} {'Type':<5} {'Chance':>10} {'Note':<25}")
csv = pd.read_csv(HCP_SUBJECTS_CSV)
csv["Subject"] = csv["Subject"].astype(str)
csv = csv.set_index("Subject")
_SUBJ_RE = re.compile(r"subject_(\d+)")
subject_dirs = sorted(HCP_ROOT.glob("subject_*"))
subject_ids = [_SUBJ_RE.search(str(d)).group(1) for d in subject_dirs]
rows = [csv.loc[s] if s in csv.index else pd.Series({c: np.nan for c in csv.columns}) for s in subject_ids]
labels_df = pd.DataFrame(rows)
for name, lcfg in HCP_LABELS.items():
    valid_idx, y = get_hcp_label(labels_df, name, labels_dict=HCP_LABELS)
    if len(y) < 50: continue
    is_clf = lcfg["type"] == "classification"
    if is_clf:
        auc = dummy_clf_auc(y, N_SPLITS)
        note = f"prev={y.mean():.3f}"
        print(f"  {name:<16} {len(y):>6} {'CLF':<5} AUC={auc:.3f}   {note:<25}")
    else:
        mae = dummy_reg_mae(y, N_SPLITS)
        note = f"mean={np.mean(y):.2f}"
        print(f"  {name:<16} {len(y):>6} {'REG':<5} MAE={mae:.3f}   {note:<25}")

# ---------- ADNI ----------
print()
print("=" * 65)
print("ADNI labels -- chance baselines")
print("=" * 65)
labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)
for name, lcfg in ADNI_LABELS.items():
    valid_idx, y = get_adni_label(labels_df, name, labels_dict=ADNI_LABELS)
    if len(y) < 50: continue
    is_clf = lcfg["type"] == "classification"
    if is_clf:
        auc = dummy_clf_auc(y, N_SPLITS)
        note = f"prev={y.mean():.3f}"
        print(f"  {name:<16} {len(y):>6} {'CLF':<5} AUC={auc:.3f}   {note:<25}")
    else:
        mae = dummy_reg_mae(y, N_SPLITS)
        note = f"mean={np.mean(y):.2f}"
        print(f"  {name:<16} {len(y):>6} {'REG':<5} MAE={mae:.3f}   {note:<25}")
