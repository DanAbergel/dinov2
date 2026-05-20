"""ADNI linear probe — minimal canonical version.

For each ADNI scan: extract the teacher's CLS token (384 dims), then
fit a single linear model per label with k-fold CV. Nothing else.

  - Classification (Sex, CDR, Degradation*) -> sklearn LogisticRegression(C=1.0)
  - Regression     (Age, MMSE)              -> sklearn RidgeCV

This matches the "linear probe" convention used in DINO v1, MoCo, SimCLR,
MAE: ONE vector per sample, ONE linear layer on top, no grid search.

Usage:
    python scripts/probe_adni.py --checkpoint outputs/.../model_*.rank_0.pth
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

# FAIR repo: label loader + ADNI label dict.
FAIR_REPO = Path(os.environ.get(
    "FAIR_DIR", "/sci/labs/arieljaffe/dan.abergel1/repos/FAIR"))
sys.path.insert(0, str(FAIR_REPO))
from src.config import (                                                # noqa: E402
    ADNI_INDEX_JSON, ADNI_LABELS, ADNI_LABELS_JSON, ADNI_ROOT,
    N_SPLITS, RANDOM_STATE,
)
from src.baselines.utils import load_adni_labels, get_label_array        # noqa: E402

# Official DINOv2 model builder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov2.models import build_model_from_cfg                           # noqa: E402
from dinov2.configs import dinov2_default_config                         # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402

ADNI_4D_PT = ADNI_ROOT / "all_4d_downsampled.pt"


# ---------- Checkpoint loading ----------

def load_teacher_backbone(checkpoint_path, cfg, device):
    """Build the teacher DinoVisionTransformer and load its weights from a
    PeriodicCheckpointer .rank_0.pth file."""
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_state = data["model"]

    # Find the 'teacher.backbone.*' subtree (FSDP wrapping may prepend
    # '_fsdp_wrapped_module.' once or twice).
    prefixes = [
        "teacher.backbone.",
        "teacher._fsdp_wrapped_module.backbone._fsdp_wrapped_module.",
        "teacher._fsdp_wrapped_module.backbone.",
    ]
    prefix = next(p for p in prefixes if any(k.startswith(p) for k in full_state))
    sd = {k[len(prefix):].replace("_fsdp_wrapped_module.", ""): v
          for k, v in full_state.items() if k.startswith(prefix)}

    _, teacher, _ = build_model_from_cfg(cfg)
    teacher.load_state_dict(sd, strict=False)
    teacher.to(device).eval()
    return teacher, data.get("iteration", -1)


def resize_pos_temporal(backbone, new_T_eff):
    """ADNI has T=140 != HCP training T=1200, so resize pos_temporal from
    (1, 120, D) to (1, T_eff_new, D) via 1D linear interpolation."""
    pe = backbone.patch_embed
    if pe.pos_temporal.shape[1] == new_T_eff:
        return
    new = F.interpolate(
        pe.pos_temporal.data.permute(0, 2, 1), size=new_T_eff,
        mode="linear", align_corners=False,
    ).permute(0, 2, 1)
    pe.pos_temporal = torch.nn.Parameter(new, requires_grad=False)
    pe.num_temporal_patches = new_T_eff
    pe.num_patches = new_T_eff * pe.num_spatial_patches


# ---------- Feature extraction (CLS only) ----------

@torch.no_grad()
def extract_cls(backbone, device, target_shape, temporal_kernel):
    """All ADNI scans -> (N, 384) numpy array of teacher CLS tokens."""
    data = torch.load(ADNI_4D_PT, weights_only=True, map_location="cpu", mmap=True)
    # Detect T axis: ADNI is either (N, X, Y, Z, T) or (N, T, X, Y, Z).
    if data.ndim == 5 and data.shape[-1] > data.shape[1]:
        layout, T, X, Y, Z = "NXYZT", data.shape[-1], *data.shape[1:4]
    else:
        layout, T, X, Y, Z = "NTXYZ", data.shape[1], *data.shape[2:5]
    N = data.shape[0]
    T = (T // temporal_kernel) * temporal_kernel  # trim to multiple of kernel
    resize_pos_temporal(backbone, T // temporal_kernel)
    print(f"  ADNI: N={N}, T={T}, spatial=({X},{Y},{Z})  -> target {target_shape}")

    embed_dim = backbone.embed_dim
    embeddings = np.empty((N, embed_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        if layout == "NXYZT":
            scan = data[i, :, :, :, :T].permute(3, 0, 1, 2).contiguous().float()
        else:
            scan = data[i, :T].contiguous().float()
        scan = scan.unsqueeze(1)  # (T, 1, X, Y, Z)
        if (X, Y, Z) != tuple(target_shape):
            scan = F.interpolate(scan, size=tuple(target_shape),
                                 mode="trilinear", align_corners=False)
        # Per-frame z-score (matches training preprocessing).
        m, s = scan.mean(dim=(1,2,3,4), keepdim=True), scan.std(dim=(1,2,3,4), keepdim=True)
        scan = torch.where(s > 1e-6, (scan - m) / s.clamp_min(1e-6), torch.zeros_like(scan))
        x = scan.unsqueeze(0).to(device, non_blocking=True)  # (1, T, 1, X, Y, Z)
        out = backbone.forward_features(x)
        embeddings[i] = out["x_norm_clstoken"].squeeze(0).cpu().numpy()
        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"    {i+1}/{N}  ({time.time() - t0:.0f}s)")
    return embeddings


# ---------- Linear probe per label ----------

def probe(X, y, groups, is_clf, n_splits):
    """K-fold CV: per fold, fit one linear model on training, evaluate on val."""
    y_cv = y.astype(np.int64) if is_clf else y.astype(np.float64)
    cv = StratifiedGroupKFold(n_splits) if is_clf else GroupKFold(n_splits)
    metrics = {}
    for tr, val in cv.split(X, y_cv, groups=groups):
        if is_clf:
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y_cv[tr])
            proba = clf.predict_proba(X[val])
            pred = clf.predict(X[val])
            m = {
                "Acc": accuracy_score(y_cv[val], pred),
                "F1":  f1_score(y_cv[val], pred, average="binary" if proba.shape[1] == 2 else "macro"),
            }
            if proba.shape[1] == 2:
                m["AUC"] = roc_auc_score(y_cv[val], proba[:, 1])
        else:
            reg = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000]).fit(X[tr], y_cv[tr])
            m = {"MAE": mean_absolute_error(y_cv[val], reg.predict(X[val]))}
        for k, v in m.items():
            metrics.setdefault(k, []).append(v)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "fold_scores": list(map(float, v))} for k, v in metrics.items()}


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-file", default="dinov2/configs/train/fmri_vits.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--features-out", default=None)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config),
                           OmegaConf.load(args.config_file))
    backbone, it = load_teacher_backbone(args.checkpoint, cfg, device)
    print(f"  Loaded teacher backbone @ iter {it}")

    X = extract_cls(backbone, device,
                    target_shape=tuple(cfg.student.fmri_img_size),
                    temporal_kernel=int(cfg.student.fmri_temporal_kernel))
    print(f"  Features: shape {X.shape}")

    if args.features_out:
        Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.features_out, features=X)

    labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)
    print(f"\n{'='*60}\nLinear probe (CLS only, k-fold={args.n_splits}) @ iter {it}\n{'='*60}")
    results = []
    for label_name, lcfg in ADNI_LABELS.items():
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=ADNI_LABELS)
        groups = labels_df.iloc[valid_idx]["subject_id"].values
        if len(y) < 50 or len(set(groups)) < args.n_splits:
            print(f"  {label_name:<16} n={len(y)}  (skipped)")
            continue
        is_clf = lcfg["type"] == "classification"
        summary = probe(X[valid_idx], y, groups, is_clf, args.n_splits)
        s = summary
        if is_clf and "AUC" in s:
            print(f"  {label_name:<16} AUC {s['AUC']['mean']:.3f}+/-{s['AUC']['std']:.3f}  Acc {s['Acc']['mean']:.3f}  F1 {s['F1']['mean']:.3f}  (n={len(y)})")
        elif is_clf:
            print(f"  {label_name:<16} Acc {s['Acc']['mean']:.3f}  F1 {s['F1']['mean']:.3f}  (n={len(y)})")
        else:
            print(f"  {label_name:<16} MAE {s['MAE']['mean']:.3f}+/-{s['MAE']['std']:.3f}  (n={len(y)})")
        results.append({"label": label_name, "is_classification": is_clf, "n": int(len(y)), "metrics": summary})

    if args.output is None:
        args.output = str(Path(args.checkpoint).parent.parent / "probes" / f"probe_iter{it:07d}.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": {"dataset": "ADNI", "checkpoint": str(args.checkpoint),
                       "iteration": int(it), "n_splits": args.n_splits,
                       "probe": "CLS + LogisticRegression(C=1.0) / RidgeCV",
                       "feature_dim": int(X.shape[1]), "seed": RANDOM_STATE},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
