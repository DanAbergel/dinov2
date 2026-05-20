"""HCP linear probe on a DINOv2-fmri checkpoint.

Same pipeline as `scripts/probe_adni.py` (official `ModelWithIntermediateLayers`
+ `LinearClassifier` + SGD cosine LR for classification, sklearn `Ridge` for
regression), evaluated on HCP labels (Sex / Age / BrainVol / GrayMatterVol /
FluidIntel / ProcSpeed / WorkingMem).

Why HCP probe is useful even though we trained on HCP:
  - SSL pretraining does NOT see labels — only scans. K-fold CV on a held-out
    subset of subjects gives valid generalisation metrics.
  - HCP labels (Sex, Age) provide a strong sanity-check baseline.
  - Cognitive labels (FluidIntel, ProcSpeed, WorkingMem) are a more demanding
    test than demographics — if they show signal, the SSL is capturing fine
    behaviour-related dynamics.

What's different from probe_adni.py:
  - **No pos_temporal resize**: HCP is T=1200 (same as training) so the
    backbone loads as-is.
  - **No spatial resize**: HCP is (45, 54, 45) same as training.
  - Data comes from `HCPFullScanDataset` (reconstructs full T=1200 scan from
    `windows.pt`), not `all_4d_downsampled.pt`.
  - Labels come from `HCP_SUBJECTS_CSV` (one row per subject), keyed by the
    subject_id parsed from each `windows.pt` path.
  - 1 scan per subject -> plain `StratifiedKFold` (classification) /
    `KFold` (regression). No GroupKFold needed.

Usage:
    python scripts/probe_hcp.py \
        --checkpoint outputs/dinov2_fmri_<ts>/model_0002999.rank_0.pth \
        --output outputs/probes/probe_hcp_iter2999.json
"""

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

# FAIR repo provides the label loader.
FAIR_REPO = Path(os.environ.get(
    "FAIR_DIR", "/sci/labs/arieljaffe/dan.abergel1/repos/FAIR"))
sys.path.insert(0, str(FAIR_REPO))

from src.config import (                                                # noqa: E402
    HCP_LABELS, HCP_ROOT, HCP_SUBJECTS_CSV, N_SPLITS, RANDOM_STATE,
)
from src.baselines.utils import get_label_array                         # noqa: E402

# OFFICIAL DINOv2 components.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov2.models import build_model_from_cfg                          # noqa: E402
from dinov2.configs import dinov2_default_config                        # noqa: E402
from dinov2.data.fmri_data import HCPFullScanDataset                    # noqa: E402
from dinov2.eval.utils import ModelWithIntermediateLayers               # noqa: E402
from dinov2.eval.linear import LinearClassifier, create_linear_input    # noqa: E402
from omegaconf import OmegaConf                                         # noqa: E402


# Official linear-probe defaults — see probe_adni.py for explanation.
N_LAST_BLOCKS = 4
USE_AVGPOOL = True
PROBE_LR_BASE = 1.0e-3
PROBE_EPOCHS = 100
PROBE_BATCH_SIZE = 32
PROBE_MOMENTUM = 0.9
PROBE_WEIGHT_DECAY = 0.0


# =============================================================================
# Checkpoint loading (identical to probe_adni.py)
# =============================================================================

def _strip_fsdp_prefix(sd):
    return {k.replace("_fsdp_wrapped_module.", ""): v for k, v in sd.items()}


def _extract_teacher_backbone_state(full_state):
    prefix_candidates = [
        "teacher.backbone.",
        "teacher._fsdp_wrapped_module.backbone._fsdp_wrapped_module.",
        "teacher._fsdp_wrapped_module.backbone.",
    ]
    matched_prefix = None
    for p in prefix_candidates:
        if any(k.startswith(p) for k in full_state):
            matched_prefix = p
            break
    if matched_prefix is None:
        raise KeyError(f"teacher.backbone not in checkpoint. Keys: {list(full_state)[:10]}")
    sd = {k[len(matched_prefix):]: v for k, v in full_state.items()
          if k.startswith(matched_prefix)}
    return _strip_fsdp_prefix(sd)


def load_teacher_backbone(checkpoint_path, train_cfg, device):
    print(f"  Loading checkpoint {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher_state = _extract_teacher_backbone_state(data["model"])
    _, teacher, embed_dim = build_model_from_cfg(train_cfg)
    missing, unexpected = teacher.load_state_dict(teacher_state, strict=False)
    if unexpected:
        print(f"  WARN: unexpected keys: {unexpected[:5]}"
              + (" ..." if len(unexpected) > 5 else ""))
    critical_missing = [k for k in (missing or []) if k != "pos_embed"]
    if critical_missing:
        print(f"  WARN: missing keys: {critical_missing[:5]}"
              + (" ..." if len(critical_missing) > 5 else ""))
    teacher.to(device).eval()
    iteration = data.get("iteration", -1)
    print(f"  Loaded teacher backbone @ iter {iteration}, embed_dim={embed_dim}")
    return teacher, iteration, embed_dim


# =============================================================================
# HCP-specific: feature extraction from full HCP scans
# =============================================================================

_SUBJECT_ID_RE = re.compile(r"subject_(\d+)")


def _path_to_subject_id(path):
    m = _SUBJECT_ID_RE.search(str(path))
    if not m:
        raise ValueError(f"Cannot parse subject_id from {path}")
    return m.group(1)


@torch.no_grad()
def extract_features_hcp(backbone, device):
    """Return (subject_ids, features) for every HCP subject in HCP_ROOT.

    Uses the same `HCPFullScanDataset` as training -> full T=1200 scans,
    z-scored per frame, shape (T, 1, X, Y, Z). No pos_temporal/spatial
    resize needed because HCP probe == HCP train distribution.
    """
    ds = HCPFullScanDataset(root=str(HCP_ROOT))
    N = len(ds)
    subject_ids = [_path_to_subject_id(p) for p in ds.paths]
    print(f"  Found {N} HCP subjects")

    wrapper = ModelWithIntermediateLayers(
        feature_model=backbone, n_last_blocks=N_LAST_BLOCKS,
        autocast_ctx=nullcontext,
    )

    # Discover feature dim with one sample.
    probe_scan = ds._load(0)                                            # (T, 1, X, Y, Z)
    sample_in = probe_scan.unsqueeze(0).to(device)                      # (1, T, 1, X, Y, Z)
    sample_out = wrapper(sample_in)
    feature_dim = create_linear_input(sample_out, N_LAST_BLOCKS, USE_AVGPOOL).shape[1]
    print(f"  Feature dim: {feature_dim} (n_blocks={N_LAST_BLOCKS}, avgpool={USE_AVGPOOL})")
    del sample_in, sample_out

    features = np.empty((N, feature_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        scan = ds._load(i).to(device, non_blocking=True).unsqueeze(0)   # (1, T, 1, X, Y, Z)
        out = wrapper(scan)
        feat = create_linear_input(out, N_LAST_BLOCKS, USE_AVGPOOL)
        features[i] = feat.squeeze(0).cpu().numpy()
        del scan, out, feat
        if (i + 1) % 25 == 0 or i == N - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{N}  ({elapsed:.0f}s, {elapsed/(i+1)*1000:.0f} ms/scan)")
    return subject_ids, features


def _build_labels_df(subject_ids):
    """Build a labels DataFrame indexed by subject_id (row order matches features)."""
    csv = pd.read_csv(HCP_SUBJECTS_CSV)
    csv["Subject"] = csv["Subject"].astype(str)
    csv = csv.set_index("Subject")
    rows = []
    for sid in subject_ids:
        if sid in csv.index:
            rows.append(csv.loc[sid])
        else:
            rows.append(pd.Series({c: np.nan for c in csv.columns}))
    return pd.DataFrame(rows)


# =============================================================================
# Per-fold training (mirrors probe_adni.py)
# =============================================================================

def _train_one_fold(X_train, y_train, X_val, y_val, *,
                    feature_dim, is_classification, device):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    if not is_classification:
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).squeeze(-1)
        ridge = Ridge(alpha=1.0).fit(X_train_s, y_train_s)
        preds_s = ridge.predict(X_val_s)
        preds = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).squeeze(-1)
        return {"MAE": mean_absolute_error(y_val.astype(np.float32), preds)}

    num_classes = int(y_train.max()) + 1
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_train_t = torch.from_numpy(X_train_s)
    X_val_t = torch.from_numpy(X_val_s).to(device)
    criterion = nn.CrossEntropyLoss()

    classifier = LinearClassifier(
        out_dim=feature_dim, use_n_blocks=N_LAST_BLOCKS,
        use_avgpool=USE_AVGPOOL, num_classes=num_classes,
    ).to(device)
    classifier.forward = lambda feat: classifier.linear(feat)           # type: ignore

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=PROBE_LR_BASE,
        momentum=PROBE_MOMENTUM, weight_decay=PROBE_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PROBE_EPOCHS,
    )

    n = X_train_t.shape[0]
    classifier.train()
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, PROBE_BATCH_SIZE):
            idx = perm[start:start + PROBE_BATCH_SIZE]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            loss = criterion(classifier(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(X_val_t).cpu()
        probs = F.softmax(val_logits, dim=-1)[:, 1].numpy() if num_classes == 2 \
            else F.softmax(val_logits, dim=-1).numpy()
        preds = val_logits.argmax(dim=-1).numpy()
    y_val_np = y_val.astype(np.int64)
    avg = "binary" if num_classes == 2 else "macro"
    metrics = {
        "Acc":  accuracy_score(y_val_np, preds),
        "F1":   f1_score(y_val_np, preds, average=avg),
        "Prec": precision_score(y_val_np, preds, average=avg, zero_division=0),
        "Rec":  recall_score(y_val_np, preds, average=avg, zero_division=0),
    }
    if num_classes == 2:
        metrics["AUC"] = roc_auc_score(y_val_np, probs)
    return metrics


def probe_one_label(X, y, label_name, is_clf, n_splits, feature_dim, device):
    y_cv = y.astype(np.int64) if is_clf else y.astype(np.float64)
    cv = (StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
          if is_clf
          else KFold(n_splits, shuffle=True, random_state=RANDOM_STATE))
    splits = cv.split(X, y_cv) if is_clf else cv.split(X)

    fold_metrics = {}
    for tr, val in splits:
        m = _train_one_fold(
            X[tr], y_cv[tr], X[val], y_cv[val],
            feature_dim=feature_dim, is_classification=is_clf, device=device,
        )
        for k, v in m.items():
            fold_metrics.setdefault(k, []).append(v)

    summary = {k: {"mean": float(np.mean(vs)), "std": float(np.std(vs)),
                   "fold_scores": list(map(float, vs))}
               for k, vs in fold_metrics.items()}
    if is_clf:
        s = summary
        auc = f"AUC {s['AUC']['mean']:.3f}+/-{s['AUC']['std']:.3f}  " if 'AUC' in s else ""
        print(f"  {label_name:<16} {auc}Acc {s['Acc']['mean']:.3f}  F1 {s['F1']['mean']:.3f}  "
              f"(n={len(y)})")
    else:
        s = summary["MAE"]
        print(f"  {label_name:<16} MAE {s['mean']:.3f}+/-{s['std']:.3f}  (n={len(y)})")
    return {"label": label_name, "is_classification": bool(is_clf),
            "n": int(len(y)), "metrics": summary}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_<iter>.rank_0.pth")
    parser.add_argument("--config-file", default="dinov2/configs/train/fmri_vits.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--features-out", default=None,
                        help="Optional .npz cache for per-subject features")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = OmegaConf.merge(
        OmegaConf.create(dinov2_default_config),
        OmegaConf.load(args.config_file),
    )
    backbone, iteration, _ = load_teacher_backbone(args.checkpoint, cfg, device)

    subject_ids, features = extract_features_hcp(backbone, device)
    feature_dim = features.shape[1]

    if args.features_out:
        cache = Path(args.features_out)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, features=features, subject_ids=np.array(subject_ids))
        print(f"Saved features: {cache}")

    labels_df = _build_labels_df(subject_ids)
    print(f"\n{'='*60}")
    print(f"OFFICIAL DINOv2 linear probe (HCP, multi-block + SGD cosine LR)")
    print(f"  Checkpoint iter: {iteration}")
    print(f"  Feature dim:     {feature_dim} (n_blocks={N_LAST_BLOCKS}, avgpool={USE_AVGPOOL})")
    print(f"  k-fold:          {args.n_splits}")
    print(f"{'='*60}")

    results = []
    for label_name, lcfg in HCP_LABELS.items():
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=HCP_LABELS)
        if len(y) < 50:
            print(f"  {label_name:<16} n={len(y)}  (skipped)")
            continue
        results.append(probe_one_label(
            X=features[valid_idx], y=y, label_name=label_name,
            is_clf=lcfg["type"] == "classification",
            n_splits=args.n_splits, feature_dim=feature_dim, device=device,
        ))

    if args.output is None:
        out_dir = Path(args.checkpoint).parent.parent / "probes"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"probe_hcp_iter{iteration:07d}.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump({
            "config": {
                "dataset": "HCP",
                "checkpoint": str(args.checkpoint),
                "iteration": int(iteration),
                "training_config": args.config_file,
                "n_splits": args.n_splits,
                "n_last_blocks": N_LAST_BLOCKS,
                "use_avgpool": USE_AVGPOOL,
                "feature_dim": feature_dim,
                "probe_lr": PROBE_LR_BASE,
                "probe_epochs": PROBE_EPOCHS,
                "probe_batch_size": PROBE_BATCH_SIZE,
                "seed": RANDOM_STATE,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
