"""ADNI non-linear probe — compares a LINEAR head vs a small MLP head on the
SAME frozen CLS features, to test whether non-linearity unlocks more clinical
signal than the linear probe (which we found plateaued).

Both heads are evaluated identically:
  - repeated StratifiedGroupKFold (n_splits x n_repeats)
  - minority oversampling in the TRAIN fold only (MLPClassifier has no
    class_weight; LogReg uses class_weight='balanced')
  - AUPRC (headline), AUC, balanced-acc; prevalence baseline reported

Heads:
  linear : balanced LogisticRegressionCV over C (= the clinical probe head)
  mlp    : StandardScaler -> MLPClassifier(hidden=(64,)), alpha picked by
           inner CV on average_precision, early stopping, on oversampled train

Feature source (one of):
  --features-in  cached .npz with 'features' (FAST, no GPU)
  --checkpoint   model_*.rank_0.pth (extract fresh; also caches to --features-out)

Usage:
  python scripts/probe_adni_nonlinear.py \
      --checkpoint outputs/dinov2_fmri_20260524_144327/model_0008999.rank_0.pth \
      --features-out outputs/probes/features_clin_144327_8999.npz
  # then iterate fast on cached features:
  python scripts/probe_adni_nonlinear.py \
      --features-in outputs/probes/features_clin_144327_8999.npz
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Self-contained labels module (was in FAIR repo; now lives here).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_labels import (                                              # noqa: E402
    ADNI_INDEX_JSON, ADNI_LABELS, ADNI_LABELS_JSON, N_SPLITS, RANDOM_STATE,
    load_adni_labels, get_label_array,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_adni import load_teacher_backbone, extract_cls                # noqa: E402
from dinov2.configs import dinov2_default_config                         # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
MLP_ALPHA_GRID = [1e-1, 1.0, 1e1]
MLP_HIDDEN = (64,)


def oversample(X, y, rng):
    """Random oversample the minority class to match the majority count."""
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts.max()
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) < n_max:
            extra = rng.choice(ci, size=n_max - len(ci), replace=True)
            ci = np.concatenate([ci, extra])
        idx.append(ci)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_linear(Xtr, ytr):
    return LogisticRegressionCV(
        Cs=C_GRID, class_weight="balanced", scoring="average_precision",
        max_iter=2000, cv=3,
    ).fit(Xtr, ytr)


def fit_mlp(Xtr, ytr, rng):
    Xos, yos = oversample(Xtr, ytr, rng)
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=500,
                      early_stopping=True, n_iter_no_change=15,
                      random_state=RANDOM_STATE),
    )
    grid = GridSearchCV(
        pipe, {"mlpclassifier__alpha": MLP_ALPHA_GRID},
        scoring="average_precision", cv=3, n_jobs=1,
    )
    grid.fit(Xos, yos)
    return grid.best_estimator_


def evaluate(clf, Xval, yval):
    proba = clf.predict_proba(Xval)[:, 1]
    pred = clf.predict(Xval)
    return (roc_auc_score(yval, proba),
            average_precision_score(yval, proba),
            balanced_accuracy_score(yval, pred))


def run_label(X, y, groups, n_splits, n_repeats):
    y = y.astype(np.int64)
    prevalence = float(y.mean())
    out = {"linear": defaultdict(list), "mlp": defaultdict(list),
           "prevalence": prevalence}
    rng = np.random.default_rng(RANDOM_STATE)
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits, shuffle=True,
                                  random_state=RANDOM_STATE + rep)
        for tr, val in cv.split(X, y, groups=groups):
            if len(np.unique(y[val])) < 2:
                continue
            for name, fitter in (("linear", lambda: fit_linear(X[tr], y[tr])),
                                  ("mlp", lambda: fit_mlp(X[tr], y[tr], rng))):
                auc, ap, ba = evaluate(fitter(), X[val], y[val])
                out[name]["AUC"].append(auc)
                out[name]["AUPRC"].append(ap)
                out[name]["BalAcc"].append(ba)
    summary = {"prevalence": prevalence}
    for head in ("linear", "mlp"):
        summary[head] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                         for k, v in out[head].items()}
    return summary


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features-in")
    src.add_argument("--checkpoint")
    ap.add_argument("--config-file", default="dinov2/configs/train/fmri_vits.yaml")
    ap.add_argument("--features-out", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--n_splits", type=int, default=N_SPLITS)
    ap.add_argument("--n_repeats", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    if args.features_in:
        X = np.load(args.features_in)["features"]
        it = -1; src_tag = os.path.basename(args.features_in)
        print(f"  Loaded cached features {X.shape}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config),
                               OmegaConf.load(args.config_file))
        backbone, it = load_teacher_backbone(args.checkpoint, cfg, device)
        X = extract_cls(backbone, device,
                        target_shape=tuple(cfg.student.fmri_img_size),
                        temporal_kernel=int(cfg.student.fmri_temporal_kernel))
        src_tag = os.path.basename(args.checkpoint)
        if args.features_out:
            Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.features_out, features=X)
            print(f"  Cached features -> {args.features_out}")

    labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)
    print(f"\n{'='*72}")
    print(f"  ADNI linear-vs-MLP probe  ({args.n_splits}-fold x {args.n_repeats})")
    print(f"  source: {src_tag}")
    print(f"{'='*72}")
    print(f"  {'label':<15} {'head':<7} {'AUPRC':>14} {'AUC':>14}  base")

    results = []
    for label_name, lcfg in ADNI_LABELS.items():
        if lcfg["type"] != "classification":
            continue
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=ADNI_LABELS)
        groups = labels_df.iloc[valid_idx]["subject_id"].values
        if len(y) < 50 or len(set(groups)) < args.n_splits:
            continue
        s = run_label(X[valid_idx], y, groups, args.n_splits, args.n_repeats)
        base = s["prevalence"]
        for head in ("linear", "mlp"):
            ap_m, ap_s = s[head]["AUPRC"]["mean"], s[head]["AUPRC"]["std"]
            au_m, au_s = s[head]["AUC"]["mean"], s[head]["AUC"]["std"]
            flag = " <-- beats base" if ap_m > base + 0.05 else ""
            print(f"  {label_name:<15} {head:<7} {ap_m:.3f}+/-{ap_s:.3f}  "
                  f"{au_m:.3f}+/-{au_s:.3f}  {base:.2f}{flag}")
        # Does MLP beat linear on AUPRC?
        delta = s["mlp"]["AUPRC"]["mean"] - s["linear"]["AUPRC"]["mean"]
        print(f"  {'':15} {'==>':<7} MLP - linear AUPRC = {delta:+.3f}")
        results.append({"label": label_name, "n": int(len(y)), "metrics": s})

    if args.output is None:
        tag = f"iter{it:07d}" if it >= 0 else Path(src_tag).stem
        Path("outputs/probes").mkdir(parents=True, exist_ok=True)
        args.output = f"outputs/probes/nonlinear_{tag}.json"
    with open(args.output, "w") as f:
        json.dump({"config": {"dataset": "ADNI", "source": src_tag,
                              "iteration": int(it), "n_splits": args.n_splits,
                              "n_repeats": args.n_repeats,
                              "heads": ["linear LogRegCV(balanced)",
                                        f"MLP{MLP_HIDDEN} + oversample"],
                              "feature_dim": int(X.shape[1]), "seed": RANDOM_STATE},
                   "results": results}, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
