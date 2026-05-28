"""ADNI CLINICAL probe — tuned for imbalanced clinical labels.

Differs from the canonical probe_adni.py (which uses a fixed
LogisticRegression(C=1.0) with no class weighting) in ways that matter
when the positive class is rare (~18% for Degradation1Y, ~145 cases):

  1. class_weight='balanced'         -> compensates the imbalance
  2. LogisticRegressionCV over C      -> inner-CV picks regularisation,
     scored by average_precision (AUPRC), per outer fold
  3. AUPRC (average_precision_score)  -> the right headline metric for
     imbalanced data; AUC alone is optimistic and high-variance here
  4. RepeatedStratifiedGroupKFold     -> n_splits x n_repeats evaluations
     to shrink the +/-0.10 fold variance we saw with plain 5-fold
  5. Reports the AUPRC baseline (= positive prevalence) so you can see
     whether the model beats "always predict the base rate"

Feature source (one of):
  --features-in  cached .npz from a canonical probe run (FAST, no GPU)
  --checkpoint   a model_*.rank_0.pth (extracts CLS fresh, needs GPU)

Usage:
  # Fast path on cached features:
  python scripts/probe_adni_clinical.py \
      --features-in outputs/probes/features_iter0008999.npz

  # Fresh extraction from a checkpoint:
  python scripts/probe_adni_clinical.py \
      --checkpoint outputs/dinov2_fmri_20260524_144327/model_0008999.rank_0.pth
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

# FAIR repo: label loader + ADNI label dict.
FAIR_REPO = Path(os.environ.get(
    "FAIR_DIR", "/sci/labs/arieljaffe/dan.abergel1/repos/FAIR"))
sys.path.insert(0, str(FAIR_REPO))
from src.config import (                                                # noqa: E402
    ADNI_INDEX_JSON, ADNI_LABELS, ADNI_LABELS_JSON,
    N_SPLITS, RANDOM_STATE,
)
from src.baselines.utils import load_adni_labels, get_label_array        # noqa: E402

# Reuse the canonical probe's feature extraction + checkpoint loading so we
# never duplicate the (subtle) pos_temporal-resize / layout-detection logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_adni import load_teacher_backbone, extract_cls                # noqa: E402
from dinov2.configs import dinov2_default_config                         # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def clinical_probe(X, y, groups, n_splits, n_repeats):
    """Repeated StratifiedGroupKFold; per fold a balanced LogisticRegressionCV
    (C chosen by inner CV on AUPRC). Returns AUC, AUPRC, balanced-acc summaries
    plus the AUPRC baseline (positive prevalence)."""
    y = y.astype(np.int64)
    prevalence = float(y.mean())
    metrics = defaultdict(list)
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits, shuffle=True,
                                  random_state=RANDOM_STATE + rep)
        for tr, val in cv.split(X, y, groups=groups):
            # Skip degenerate folds (one class absent in val).
            if len(np.unique(y[val])) < 2:
                continue
            clf = LogisticRegressionCV(
                Cs=C_GRID, class_weight="balanced", scoring="average_precision",
                max_iter=2000, cv=3,
            ).fit(X[tr], y[tr])
            proba = clf.predict_proba(X[val])[:, 1]
            pred = clf.predict(X[val])
            metrics["AUC"].append(roc_auc_score(y[val], proba))
            metrics["AUPRC"].append(average_precision_score(y[val], proba))
            metrics["BalAcc"].append(balanced_accuracy_score(y[val], pred))
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "n_folds": len(v)} for k, v in metrics.items()}
    summary["prevalence"] = prevalence            # AUPRC baseline
    return summary


def reg_probe(X, y, groups, n_splits, n_repeats):
    y = y.astype(np.float64)
    maes = []
    for rep in range(n_repeats):
        cv = GroupKFold(n_splits)
        for tr, val in cv.split(X, y, groups=groups):
            reg = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000]).fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[val], reg.predict(X[val])))
    return {"MAE": {"mean": float(np.mean(maes)), "std": float(np.std(maes)),
                    "n_folds": len(maes)}}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features-in", help="cached .npz with 'features' array")
    src.add_argument("--checkpoint", help="model_*.rank_0.pth (extract fresh)")
    ap.add_argument("--config-file", default="dinov2/configs/train/fmri_vits.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--n_splits", type=int, default=N_SPLITS)
    ap.add_argument("--n_repeats", type=int, default=4,
                    help="repeats of the k-fold split (variance reduction)")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    # ----- Feature source -----
    if args.features_in:
        X = np.load(args.features_in)["features"]
        it = -1
        src_tag = os.path.basename(args.features_in)
        print(f"  Loaded cached features {X.shape} from {args.features_in}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config),
                               OmegaConf.load(args.config_file))
        backbone, it = load_teacher_backbone(args.checkpoint, cfg, device)
        X = extract_cls(backbone, device,
                        target_shape=tuple(cfg.student.fmri_img_size),
                        temporal_kernel=int(cfg.student.fmri_temporal_kernel))
        src_tag = os.path.basename(args.checkpoint)
        print(f"  Extracted features {X.shape} from {args.checkpoint} @ iter {it}")

    labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)
    print(f"\n{'='*68}")
    print(f"  ADNI CLINICAL probe  (balanced LogRegCV, {args.n_splits}-fold "
          f"x {args.n_repeats} repeats)")
    print(f"  source: {src_tag}")
    print(f"{'='*68}")
    print(f"  {'label':<16} {'AUPRC':>16} {'AUC':>14} {'BalAcc':>10}  base  n")

    results = []
    for label_name, lcfg in ADNI_LABELS.items():
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=ADNI_LABELS)
        groups = labels_df.iloc[valid_idx]["subject_id"].values
        if len(y) < 50 or len(set(groups)) < args.n_splits:
            continue
        is_clf = lcfg["type"] == "classification"
        if is_clf:
            s = clinical_probe(X[valid_idx], y, groups, args.n_splits, args.n_repeats)
            ap_m, ap_s = s["AUPRC"]["mean"], s["AUPRC"]["std"]
            auc_m, auc_s = s["AUC"]["mean"], s["AUC"]["std"]
            ba = s["BalAcc"]["mean"]
            base = s["prevalence"]
            # Flag labels where AUPRC clearly beats the base rate.
            flag = " <-- beats base" if ap_m > base + 0.05 else ""
            print(f"  {label_name:<16} {ap_m:.3f}+/-{ap_s:.3f}  "
                  f"{auc_m:.3f}+/-{auc_s:.3f}  {ba:.3f}  {base:.2f}  {len(y)}{flag}")
        else:
            s = reg_probe(X[valid_idx], y, groups, args.n_splits, args.n_repeats)
            print(f"  {label_name:<16} MAE {s['MAE']['mean']:.3f}+/-{s['MAE']['std']:.3f}"
                  f"  (n={len(y)})")
        results.append({"label": label_name, "is_classification": is_clf,
                        "n": int(len(y)), "metrics": s})

    # ----- Save -----
    if args.output is None:
        tag = f"iter{it:07d}" if it >= 0 else Path(src_tag).stem
        outdir = Path("outputs/probes")
        outdir.mkdir(parents=True, exist_ok=True)
        args.output = str(outdir / f"clinical_{tag}.json")
    with open(args.output, "w") as f:
        json.dump({
            "config": {"dataset": "ADNI", "source": src_tag, "iteration": int(it),
                       "n_splits": args.n_splits, "n_repeats": args.n_repeats,
                       "probe": "CLS + balanced LogisticRegressionCV(AUPRC) / RidgeCV",
                       "C_grid": C_GRID, "feature_dim": int(X.shape[1]),
                       "seed": RANDOM_STATE},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
