"""HCP linear probe — minimal canonical version.

For each HCP scan: extract the teacher's CLS token (384 dims), then
fit a single linear model per label with k-fold CV. Nothing else.

Same simplicity as probe_adni.py: one CLS vector per sample, one linear
model (LogisticRegression for classification, RidgeCV for regression).

HCP has T=1200. If the model was trained at a different T (e.g. Mixed
HCP+ADNI at T=140 -> T_eff=10), the pos_temporal table is resized via
1D linear interpolation to match HCP's T_eff at inference time. Same
logic as probe_adni.py.

Usage:
    python scripts/probe_hcp.py --checkpoint outputs/.../model_*.rank_0.pth
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold

# Self-contained labels module (was in FAIR repo; now lives here).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_labels import (                                              # noqa: E402
    HCP_LABELS, HCP_ROOT, HCP_SUBJECTS_CSV, N_SPLITS, RANDOM_STATE,
    get_label_array,
)

# Official DINOv2 model builder + HCP dataset.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov2.models import build_model_from_cfg                           # noqa: E402
from dinov2.configs import dinov2_default_config                         # noqa: E402
from dinov2.data.fmri_data import HCPFullScanDataset                     # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402

_SUBJECT_RE = re.compile(r"subject_(\d+)")


# ---------- Checkpoint loading ----------

def load_teacher_backbone(checkpoint_path, cfg, device):
    """Build the teacher DinoVisionTransformer and load its weights."""
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_state = data["model"]
    prefixes = [
        "teacher.backbone.",
        "teacher._fsdp_wrapped_module.backbone._fsdp_wrapped_module.",
        "teacher._fsdp_wrapped_module.backbone.",
    ]
    prefix = next(p for p in prefixes if any(k.startswith(p) for k in full_state))
    sd = {k[len(prefix):].replace("_fsdp_wrapped_module.", ""): v
          for k, v in full_state.items() if k.startswith(prefix)}
    # Backward-compat: older checkpoints stored the factorised pos embedding
    # flat (patch_embed.pos_temporal); current code nests it under
    # patch_embed.pos.* (PositionEmbedding3D submodule).
    for k in ("pos_temporal", "pos_spatial", "pos_cls"):
        flat = f"patch_embed.{k}"
        if flat in sd:
            sd[f"patch_embed.pos.{k}"] = sd.pop(flat)
    _, teacher, _ = build_model_from_cfg(cfg)
    teacher.load_state_dict(sd, strict=False)
    teacher.to(device).eval()
    return teacher, data.get("iteration", -1)


# ---------- pos_temporal resize ----------

def resize_pos_temporal(backbone, new_T_eff):
    """The model's T_eff (from training T) may differ from the probe data's
    T_eff. Resize pos_temporal via 1D linear interpolation."""
    pe = backbone.patch_embed.pos               # PositionEmbedding3D
    if pe.pos_temporal.shape[1] == new_T_eff:
        return
    new = F.interpolate(
        pe.pos_temporal.data.permute(0, 2, 1), size=new_T_eff,
        mode="linear", align_corners=False,
    ).permute(0, 2, 1)
    pe.pos_temporal = torch.nn.Parameter(new, requires_grad=False)
    pe.num_temporal_patches = new_T_eff
    backbone.patch_embed.num_temporal_patches = new_T_eff
    backbone.patch_embed.num_patches = new_T_eff * pe.num_spatial_patches
    print(f"  pos_temporal resized to T_eff={new_T_eff}")


# ---------- Feature extraction (CLS only) ----------

@torch.no_grad()
def extract_cls(backbone, device, temporal_kernel):
    """All HCP scans -> (N, 384) numpy array of teacher CLS tokens."""
    ds = HCPFullScanDataset(root=str(HCP_ROOT))
    N = len(ds)
    subject_ids = [_SUBJECT_RE.search(str(p)).group(1) for p in ds.paths]
    # Peek at the temporal length so we can size pos_temporal once.
    probe_scan = ds._load(0)
    T = probe_scan.shape[0]
    new_T_eff = T // temporal_kernel
    resize_pos_temporal(backbone, new_T_eff)
    print(f"  HCP: {N} subjects, T={T} -> T_eff={new_T_eff}")

    embed_dim = backbone.embed_dim
    embeddings = np.empty((N, embed_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        scan = ds._load(i).to(device, non_blocking=True).unsqueeze(0)  # (1, T, 1, X, Y, Z)
        out = backbone.forward_features(scan)
        embeddings[i] = out["x_norm_clstoken"].squeeze(0).cpu().numpy()
        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"    {i+1}/{N}  ({time.time() - t0:.0f}s)")
    return subject_ids, embeddings


def build_labels_df(subject_ids):
    """Lookup labels for each subject from HCP_SUBJECTS_CSV (row order matches features)."""
    csv = pd.read_csv(HCP_SUBJECTS_CSV)
    csv["Subject"] = csv["Subject"].astype(str)
    csv = csv.set_index("Subject")
    rows = [csv.loc[sid] if sid in csv.index else pd.Series({c: np.nan for c in csv.columns})
            for sid in subject_ids]
    return pd.DataFrame(rows)


# ---------- Linear probe per label ----------

def probe(X, y, is_clf, n_splits):
    """K-fold CV: per fold, fit one linear model, evaluate on val.
    HCP = 1 scan per subject -> no GroupKFold needed."""
    y_cv = y.astype(np.int64) if is_clf else y.astype(np.float64)
    cv = (StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE) if is_clf
          else KFold(n_splits, shuffle=True, random_state=RANDOM_STATE))
    metrics = {}
    for tr, val in (cv.split(X, y_cv) if is_clf else cv.split(X)):
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

    subject_ids, X = extract_cls(backbone, device,
                                  temporal_kernel=cfg.student.fmri_temporal_kernel)
    print(f"  Features: shape {X.shape}")

    if args.features_out:
        Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.features_out, features=X, subject_ids=np.array(subject_ids))

    labels_df = build_labels_df(subject_ids)
    print(f"\n{'='*60}\nLinear probe (CLS only, k-fold={args.n_splits}) @ iter {it}\n{'='*60}")
    results = []
    for label_name, lcfg in HCP_LABELS.items():
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=HCP_LABELS)
        if len(y) < 50:
            print(f"  {label_name:<16} n={len(y)}  (skipped)")
            continue
        is_clf = lcfg["type"] == "classification"
        summary = probe(X[valid_idx], y, is_clf, args.n_splits)
        s = summary
        if is_clf and "AUC" in s:
            print(f"  {label_name:<16} AUC {s['AUC']['mean']:.3f}+/-{s['AUC']['std']:.3f}  Acc {s['Acc']['mean']:.3f}  (n={len(y)})")
        elif is_clf:
            print(f"  {label_name:<16} Acc {s['Acc']['mean']:.3f}  (n={len(y)})")
        else:
            print(f"  {label_name:<16} MAE {s['MAE']['mean']:.3f}+/-{s['MAE']['std']:.3f}  (n={len(y)})")
        results.append({"label": label_name, "is_classification": is_clf, "n": int(len(y)), "metrics": summary})

    if args.output is None:
        args.output = str(Path(args.checkpoint).parent.parent / "probes" / f"probe_hcp_iter{it:07d}.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": {"dataset": "HCP", "checkpoint": str(args.checkpoint),
                       "iteration": int(it), "n_splits": args.n_splits,
                       "probe": "CLS + LogisticRegression(C=1.0) / RidgeCV",
                       "feature_dim": int(X.shape[1]), "seed": RANDOM_STATE},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
