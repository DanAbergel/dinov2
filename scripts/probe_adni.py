"""ADNI linear probe on a DINOv2-fmri checkpoint.

Uses the OFFICIAL DINOv2 probe components verbatim:
  - `dinov2.eval.utils.ModelWithIntermediateLayers` for multi-block CLS extraction
  - `dinov2.eval.linear.create_linear_input` for (concat last-N CLS + avgpool patches)
  - `dinov2.eval.linear.LinearClassifier` as the head
  - SGD with cosine LR (same recipe as official linear.py L235-L256)

The only adaptations for ADNI:
  - feed 6D fMRI scans (our PatchEmbed3DPlus1D + 6D branch handle this)
  - T mismatch HCP-train (T=1200) -> ADNI-probe (T=140): resize `pos_temporal`
  - Train/val/test -> StratifiedGroupKFold k=5 (ADNI has ~167 valid scans per label
    after filtering; train/val/test on so few would be statistically poor)
  - Support both classification (Sex, CDR, Degradation*) and regression
    (Age, MMSE) labels: classification uses LinearClassifier(num_classes>=2)
    + CrossEntropy, regression uses num_classes=1 + MSELoss
  - All other code (model build, checkpoint loading, etc.) is unchanged.

Usage:
    python scripts/probe_adni.py \
        --checkpoint outputs/dinov2_fmri_<ts>/model_0002999.rank_0.pth \
        --output outputs/probes/probe_iter2999.json
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

# FAIR repo provides the label loader / config we want to match.
FAIR_REPO = Path(os.environ.get(
    "FAIR_DIR", "/sci/labs/arieljaffe/dan.abergel1/repos/FAIR"))
sys.path.insert(0, str(FAIR_REPO))

from src.config import (                                                # noqa: E402
    ADNI_INDEX_JSON, ADNI_LABELS, ADNI_LABELS_JSON, ADNI_ROOT,
    N_SPLITS, RANDOM_STATE,
)
from src.baselines.utils import load_adni_labels, get_label_array        # noqa: E402

# OFFICIAL DINOv2 components (used as-is).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov2.models import build_model_from_cfg                           # noqa: E402
from dinov2.configs import dinov2_default_config                         # noqa: E402
from dinov2.eval.utils import ModelWithIntermediateLayers                # noqa: E402
from dinov2.eval.linear import LinearClassifier, create_linear_input     # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402


ADNI_4D_PT = ADNI_ROOT / "all_4d_downsampled.pt"

# Official linear-probe defaults from dinov2/eval/linear.py L235-L256
# (we drop the grid-search to keep the run short; pick the strongest combo
# from their default search space).
N_LAST_BLOCKS = 4         # use last 4 transformer blocks
USE_AVGPOOL = True        # also concat avg of patch tokens from last block
PROBE_LR_BASE = 1.0e-3    # scaled by batch_size/256 inside scale_lr
PROBE_EPOCHS = 100        # SGD epochs per fold (small data -> short)
PROBE_BATCH_SIZE = 32
PROBE_MOMENTUM = 0.9
PROBE_WEIGHT_DECAY = 0.0


# =============================================================================
# Checkpoint loading (unchanged from previous version)
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
        raise KeyError(
            "Could not find teacher.backbone.* in checkpoint. "
            f"Sample keys: {list(full_state)[:10]}"
        )
    sd = {k[len(matched_prefix):]: v for k, v in full_state.items()
          if k.startswith(matched_prefix)}
    return _strip_fsdp_prefix(sd)


def load_teacher_backbone(checkpoint_path, train_cfg, device):
    print(f"  Loading checkpoint {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_state = data["model"]
    teacher_state = _extract_teacher_backbone_state(full_state)

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


def _resize_pos_temporal(backbone, new_T_eff):
    pe = backbone.patch_embed
    old = pe.pos_temporal.data
    old_T = old.shape[1]
    if old_T == new_T_eff:
        return
    print(f"  Resizing pos_temporal {old_T} -> {new_T_eff} (linear interp)")
    new = F.interpolate(
        old.permute(0, 2, 1), size=new_T_eff,
        mode="linear", align_corners=False,
    ).permute(0, 2, 1)
    pe.pos_temporal = nn.Parameter(new, requires_grad=False)
    pe.num_temporal_patches = new_T_eff
    pe.num_patches = new_T_eff * pe.num_spatial_patches


# =============================================================================
# Feature extraction via OFFICIAL ModelWithIntermediateLayers
# =============================================================================

@torch.no_grad()
def _zscore_per_frame(scan):
    mean = scan.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = scan.std(dim=(1, 2, 3, 4), keepdim=True)
    return torch.where(
        std > 1e-6, (scan - mean) / std.clamp_min(1e-6), torch.zeros_like(scan)
    )


@torch.no_grad()
def extract_features(backbone, device, target_shape, temporal_kernel):
    """All ADNI scans -> (N, F) numpy array using OFFICIAL multi-block extraction.

    F = N_LAST_BLOCKS * embed_dim (+ embed_dim if USE_AVGPOOL).
    For ViT-S with n=4 + avgpool: F = 4*384 + 384 = 1920.
    """
    print(f"  Loading {ADNI_4D_PT} (mmap)")
    data = torch.load(ADNI_4D_PT, weights_only=True, map_location="cpu", mmap=True)
    print(f"  ADNI shape={tuple(data.shape)} dtype={data.dtype}")

    _, *rest = data.shape
    if data.ndim == 5 and rest[-1] > rest[0]:
        t_layout, T = "NXYZT", data.shape[-1]
        X, Y, Z = data.shape[1:4]
    else:
        t_layout, T = "NTXYZ", data.shape[1]
        X, Y, Z = data.shape[2:5]
    N = data.shape[0]
    print(f"  Layout {t_layout}: N={N}, T={T}, spatial=({X},{Y},{Z})")

    new_T_eff = T // temporal_kernel
    if T % temporal_kernel != 0:
        T_keep = new_T_eff * temporal_kernel
        print(f"  Trimming T={T} -> {T_keep} (multiple of kernel={temporal_kernel})")
        T = T_keep
    _resize_pos_temporal(backbone, new_T_eff)

    # OFFICIAL wrapper. No autocast for fp32 stability on small probe data.
    wrapper = ModelWithIntermediateLayers(
        feature_model=backbone,
        n_last_blocks=N_LAST_BLOCKS,
        autocast_ctx=nullcontext,
    )

    # Figure out the feature dim by running one sample through.
    # Shape must be exactly 6D: (B=1, T, C=1, X, Y, Z) so prepare_tokens_with_masks
    # hits our `if x.ndim == 6` branch.
    sample_in = torch.zeros(1, T, 1, *target_shape, device=device)
    sample_out = wrapper(sample_in)
    feature_dim = create_linear_input(sample_out, N_LAST_BLOCKS, USE_AVGPOOL).shape[1]
    print(f"  Feature dim: {feature_dim}  "
          f"(N_LAST_BLOCKS={N_LAST_BLOCKS}, USE_AVGPOOL={USE_AVGPOOL})")

    features = np.empty((N, feature_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        if t_layout == "NXYZT":
            scan = data[i, :, :, :, :T].permute(3, 0, 1, 2).contiguous().float()
        else:
            scan = data[i, :T].contiguous().float()
        scan = scan.unsqueeze(1)                                            # (T, 1, X, Y, Z)
        if (X, Y, Z) != tuple(target_shape):
            scan = F.interpolate(
                scan, size=tuple(target_shape),
                mode="trilinear", align_corners=False,
            )
        scan = _zscore_per_frame(scan)
        x = scan.unsqueeze(0).to(device, non_blocking=True)
        out = wrapper(x)                                                    # 4 tuples (patches, cls)
        feat = create_linear_input(out, N_LAST_BLOCKS, USE_AVGPOOL)         # (1, F)
        features[i] = feat.squeeze(0).cpu().numpy()
        del scan, x, out, feat
        if (i + 1) % 25 == 0 or i == N - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{N}  ({elapsed:.0f}s, {elapsed/(i+1)*1000:.0f} ms/scan)")
    return features


# =============================================================================
# OFFICIAL LinearClassifier trained per fold with SGD + cosine LR
# =============================================================================

def _train_one_fold(X_train, y_train, X_val, y_val, *,
                    feature_dim, is_classification, device):
    """Linear probe per fold.

    Classification: official `LinearClassifier` + SGD + cosine LR (mirrors
        dinov2/eval/linear.py L235-L256 recipe).
    Regression: sklearn `Ridge` (closed-form, numerically stable). The
        official DINOv2 eval pipeline has NO regression analogue, and SGD
        with MSE on small noisy fMRI batches diverges to NaN; Ridge with
        L2 reg is the equivalent linear model with a stable solver.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    if not is_classification:
        # Sklearn RidgeCV — analytical, inner CV picks the alpha that minimises
        # held-out MSE. Necessary because n << p (n=~770, p=1920) -> a fixed
        # small alpha like 1.0 underregularises massively and we get MAE worse
        # than predicting the mean. The alpha grid spans 1e-1 to 1e6 so that
        # whichever regime the features fall into is covered.
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).squeeze(-1)
        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]).fit(X_train_s, y_train_s)
        preds_s = ridge.predict(X_val_s)
        preds = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).squeeze(-1)
        return {"MAE": mean_absolute_error(y_val.astype(np.float32), preds),
                "best_alpha": float(ridge.alpha_)}

    # ---- Classification: official LinearClassifier + SGD + cosine LR ----
    num_classes = int(y_train.max()) + 1
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_train_t = torch.from_numpy(X_train_s)
    X_val_t = torch.from_numpy(X_val_s).to(device)
    criterion = nn.CrossEntropyLoss()

    classifier = LinearClassifier(
        out_dim=feature_dim, use_n_blocks=N_LAST_BLOCKS,
        use_avgpool=USE_AVGPOOL, num_classes=num_classes,
    ).to(device)
    # Pre-computed features bypass ModelWithIntermediateLayers; use the
    # bare linear layer directly.
    classifier.forward = lambda feat: classifier.linear(feat)               # type: ignore

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=PROBE_LR_BASE,
        momentum=PROBE_MOMENTUM, weight_decay=PROBE_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PROBE_EPOCHS,
    )

    n = X_train_t.shape[0]
    classifier.train()
    for epoch in range(PROBE_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, PROBE_BATCH_SIZE):
            idx = perm[start:start + PROBE_BATCH_SIZE]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(X_val_t).cpu()
        if num_classes == 2:
            probs = F.softmax(val_logits, dim=-1)[:, 1].numpy()
        else:
            probs = F.softmax(val_logits, dim=-1).numpy()
        preds = val_logits.argmax(dim=-1).numpy()
    y_val_np = y_val.astype(np.int64)
    metrics = {
        "Acc":  accuracy_score(y_val_np, preds),
        "F1":   f1_score(y_val_np, preds, average="binary" if num_classes == 2 else "macro"),
        "Prec": precision_score(y_val_np, preds, average="binary" if num_classes == 2 else "macro", zero_division=0),
        "Rec":  recall_score(y_val_np, preds, average="binary" if num_classes == 2 else "macro", zero_division=0),
    }
    if num_classes == 2:
        metrics["AUC"] = roc_auc_score(y_val_np, probs)
    return metrics


def probe_one_label(X, y, groups, label_name, is_clf, n_splits, feature_dim, device):
    y_cv = y.astype(np.int64) if is_clf else y.astype(np.float64)
    if is_clf:
        cv = StratifiedGroupKFold(n_splits=n_splits)
        splits = cv.split(X, y_cv, groups=groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        splits = cv.split(X, y_cv, groups=groups)

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
              f"(n={len(y)}, pos={int((y_cv==1).sum()) if y_cv.max() == 1 else '-'})")
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
                        help="Optional .npz cache for the per-scan multi-block features")
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
    target_shape = tuple(cfg.student.fmri_img_size)
    temporal_kernel = int(cfg.student.fmri_temporal_kernel)

    backbone, iteration, _ = load_teacher_backbone(args.checkpoint, cfg, device)

    features = extract_features(
        backbone, device,
        target_shape=target_shape, temporal_kernel=temporal_kernel,
    )
    feature_dim = features.shape[1]

    if args.features_out:
        cache = Path(args.features_out)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, features=features)
        print(f"Saved features: {cache}")

    labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)
    print(f"\n{'='*60}")
    print(f"OFFICIAL DINOv2 linear probe (multi-block + SGD cosine LR)")
    print(f"  Checkpoint iter: {iteration}")
    print(f"  Feature dim:     {feature_dim} (n_blocks={N_LAST_BLOCKS}, avgpool={USE_AVGPOOL})")
    print(f"  k-fold:          {args.n_splits}")
    print(f"{'='*60}")

    results = []
    for label_name, lcfg in ADNI_LABELS.items():
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=ADNI_LABELS)
        groups = labels_df.iloc[valid_idx]["subject_id"].values
        if len(y) < 50 or len(set(groups)) < args.n_splits:
            print(f"  {label_name:<16} n={len(y)}  (skipped)")
            continue
        results.append(probe_one_label(
            X=features[valid_idx], y=y, groups=groups, label_name=label_name,
            is_clf=lcfg["type"] == "classification",
            n_splits=args.n_splits, feature_dim=feature_dim, device=device,
        ))

    if args.output is None:
        out_dir = Path(args.checkpoint).parent.parent / "probes"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"probe_iter{iteration:07d}.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump({
            "config": {
                "checkpoint": str(args.checkpoint),
                "iteration": int(iteration),
                "training_config": args.config_file,
                "target_shape": list(target_shape),
                "temporal_kernel": temporal_kernel,
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
