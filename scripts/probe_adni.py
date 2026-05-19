"""ADNI linear probe on a DINOv2-fmri checkpoint.

Loads any intermediate `model_<iter>.rank_0.pth` produced by the official
PeriodicCheckpointer, extracts the TEACHER CLS embedding per ADNI scan,
then runs StratifiedGroupKFold k=5 with LBFGS LogReg / Ridge (the same
pipeline as `FAIR/src/dino/adni_probe.py:fit_linear_probe`).

Handles the T mismatch between training (HCP T=1200) and probe (ADNI
T=140) by 1D-interpolating the model's `pos_temporal` from
(1, T_eff_train, D) to (1, T_eff_probe, D) before inference. The
Conv1d temporal kernel itself is kernel-size agnostic, so applying it
to T=140 just produces T_eff=14 output frames; only the pos table needs
resampling.

Usage:
    python scripts/probe_adni.py \
        --checkpoint outputs/dinov2_fmri_<ts>/model_0011999.rank_0.pth \
        --output outputs/probes/probe_iter11999.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the FAIR repo importable so we can reuse the existing label loader
# and probe fitter (apples-to-apples comparison with prior runs).
FAIR_REPO = Path(os.environ.get(
    "FAIR_DIR", "/sci/labs/arieljaffe/dan.abergel1/repos/FAIR"))
sys.path.insert(0, str(FAIR_REPO))

from src.config import (                                                # noqa: E402
    ADNI_INDEX_JSON, ADNI_LABELS, ADNI_LABELS_JSON, ADNI_ROOT,
    N_SPLITS, RANDOM_STATE,
)
from src.baselines.utils import load_adni_labels, get_label_array        # noqa: E402
from src.dino.adni_probe import (                                        # noqa: E402
    fit_linear_probe, probe_one_label,
)

# Build the official DINOv2 backbone exactly like at training time.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov2.models import build_model_from_cfg                           # noqa: E402
from dinov2.configs import dinov2_default_config                         # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402


ADNI_4D_PT = ADNI_ROOT / "all_4d_downsampled.pt"


# =============================================================================
# Checkpoint loading
# =============================================================================

def _strip_fsdp_prefix(sd):
    """FSDP LOCAL_STATE_DICT prepends '_fsdp_wrapped_module.' to every key when
    the module is wrapped. Strip it so the keys match the plain backbone."""
    return {k.replace("_fsdp_wrapped_module.", ""): v for k, v in sd.items()}


def _extract_teacher_backbone_state(full_state):
    """Pull the teacher.backbone subtree from the saved SSLMetaArch state dict.

    The saved dict's keys look like 'teacher.backbone.<...>' (or
    'teacher._fsdp_wrapped_module.backbone.<...>' with FSDP wrapping).
    We isolate the backbone, strip the prefix, and strip FSDP markers.
    """
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
        # Fallback: dump a sample of keys for debugging.
        raise KeyError(
            "Could not find teacher.backbone.* in checkpoint. "
            f"Sample keys: {list(full_state)[:10]}"
        )
    sd = {k[len(matched_prefix):]: v for k, v in full_state.items()
          if k.startswith(matched_prefix)}
    return _strip_fsdp_prefix(sd)


def load_teacher_backbone(checkpoint_path, train_cfg, device):
    """Build the teacher DinoVisionTransformer + PatchEmbed3DPlus1D and load
    the saved state. Returns the backbone in eval mode on `device`."""
    print(f"  Loading checkpoint {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in data:
        raise KeyError(
            f"Checkpoint has no 'model' key. Top-level keys: {list(data)}"
        )
    full_state = data["model"]
    teacher_state = _extract_teacher_backbone_state(full_state)

    # Build a fresh backbone using the same config as training so the
    # parameter shapes match. `cfg.student` carries the architectural
    # spec that produced this checkpoint.
    _, teacher, embed_dim = build_model_from_cfg(train_cfg)

    # Load. We expect a perfect match for fMRI keys; if anything mismatches,
    # surface it loudly rather than silently dropping params.
    missing, unexpected = teacher.load_state_dict(teacher_state, strict=False)
    if unexpected:
        print(f"  WARN: unexpected keys in checkpoint: {unexpected[:5]}"
              + (" ..." if len(unexpected) > 5 else ""))
    if missing:
        # `pos_embed` (the unused official 2D table) is always missing
        # because we drop it in fMRI mode — that's OK.
        critical_missing = [k for k in missing if k != "pos_embed"]
        if critical_missing:
            print(f"  WARN: missing keys: {critical_missing[:5]}"
                  + (" ..." if len(critical_missing) > 5 else ""))

    teacher.to(device).eval()
    iteration = data.get("iteration", -1)
    print(f"  Loaded teacher backbone @ iter {iteration}, embed_dim={embed_dim}")
    return teacher, iteration


def _resize_pos_temporal(backbone, new_T_eff):
    """Interpolate pos_temporal from (1, T_eff_train, D) to (1, T_eff_probe, D).

    `pos_spatial` is unchanged (same spatial grid as training). The Conv1d
    kernel itself accepts any T, so only pos_temporal needs resampling.
    """
    pe = backbone.patch_embed
    old = pe.pos_temporal.data                                     # (1, T_old, D)
    old_T = old.shape[1]
    if old_T == new_T_eff:
        return
    print(f"  Resizing pos_temporal {old_T} -> {new_T_eff} (linear interp)")
    new = F.interpolate(
        old.permute(0, 2, 1),                                      # (1, D, T_old)
        size=new_T_eff, mode="linear", align_corners=False,
    ).permute(0, 2, 1)                                             # (1, T_new, D)
    pe.pos_temporal = torch.nn.Parameter(new, requires_grad=False)
    pe.num_temporal_patches = new_T_eff
    pe.num_patches = new_T_eff * pe.num_spatial_patches


# =============================================================================
# Embedding extraction
# =============================================================================

@torch.no_grad()
def _zscore_per_frame(scan):
    mean = scan.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = scan.std(dim=(1, 2, 3, 4), keepdim=True)
    return torch.where(
        std > 1e-6, (scan - mean) / std.clamp_min(1e-6), torch.zeros_like(scan)
    )


@torch.no_grad()
def extract_embeddings(backbone, device, target_shape, temporal_kernel):
    """All ADNI scans -> (N, embed_dim) numpy array of teacher CLS tokens."""
    print(f"  Loading {ADNI_4D_PT}")
    data = torch.load(ADNI_4D_PT, weights_only=True, map_location="cpu")
    print(f"  ADNI shape={tuple(data.shape)} dtype={data.dtype}")

    # Detect time axis: T is the largest non-batch dim for ADNI (~140 vs 45-55).
    _, *rest = data.shape
    if data.ndim == 5 and rest[-1] > rest[0]:
        # (N, X, Y, Z, T) -> permute to (N, T, X, Y, Z)
        data = data.permute(0, 4, 1, 2, 3).contiguous()
        print(f"  Permuted to (N, T, X, Y, Z): {tuple(data.shape)}")

    N, T, X, Y, Z = data.shape
    new_T_eff = T // temporal_kernel
    if T % temporal_kernel != 0:
        # Trim trailing frames so T is divisible by temporal_kernel.
        T_keep = new_T_eff * temporal_kernel
        print(f"  Trimming T={T} -> {T_keep} (multiple of kernel={temporal_kernel})")
        data = data[:, :T_keep]
        T = T_keep
    _resize_pos_temporal(backbone, new_T_eff)

    embeddings = np.empty((N, backbone.embed_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        scan = data[i].float().unsqueeze(1)                        # (T, 1, X, Y, Z)
        # Resize spatial if needed.
        if (X, Y, Z) != tuple(target_shape):
            scan = F.interpolate(
                scan.unsqueeze(0).permute(0, 2, 1, 3, 4, 5).reshape(1, 1, T, X, Y, Z),
                size=(T,) + tuple(target_shape),
                mode="trilinear", align_corners=False,
            ).reshape(1, 1, T, *target_shape).permute(0, 2, 1, 3, 4, 5).squeeze(0)
        scan = _zscore_per_frame(scan)
        x = scan.unsqueeze(0).to(device, non_blocking=True)        # (1, T, 1, X, Y, Z)
        out = backbone(x, is_training=True)
        embeddings[i] = out["x_norm_clstoken"].squeeze(0).cpu().numpy()
        if (i + 1) % 25 == 0 or i == N - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{N}  ({elapsed:.0f}s, {elapsed/(i+1)*1000:.0f} ms/scan)")
    return embeddings


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_<iter>.rank_0.pth")
    parser.add_argument("--config-file", default="dinov2/configs/train/fmri_vits.yaml",
                        help="The YAML used to TRAIN this checkpoint (must match the "
                             "architecture; we read student.* / fmri_* from it).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Default: outputs/probes/probe_<tag>.json")
    parser.add_argument("--embeddings-out", default=None,
                        help="Optional .npz cache for the per-scan embeddings")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reload the training config so we know the exact architecture used.
    cfg = OmegaConf.merge(
        OmegaConf.create(dinov2_default_config),
        OmegaConf.load(args.config_file),
    )
    target_shape = tuple(cfg.student.fmri_img_size)
    temporal_kernel = int(cfg.student.fmri_temporal_kernel)

    backbone, iteration = load_teacher_backbone(args.checkpoint, cfg, device)

    embeddings = extract_embeddings(
        backbone, device, target_shape=target_shape, temporal_kernel=temporal_kernel,
    )

    if args.embeddings_out:
        cache = Path(args.embeddings_out)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=embeddings)
        print(f"Saved embeddings: {cache}")

    # ---- k-fold linear probe on ADNI labels (same pipeline as FAIR/src/dino) ----
    labels_df = load_adni_labels(ADNI_INDEX_JSON, ADNI_LABELS_JSON)
    print(f"\n{'='*60}")
    print(f"Linear probe  |  DINOv2-fmri @ iter {iteration}  |  "
          f"StratifiedGroupKFold k={args.n_splits}")
    print(f"{'='*60}")

    results = []
    for label_name, lcfg in ADNI_LABELS.items():
        valid_idx, y = get_label_array(labels_df, label_name, labels_dict=ADNI_LABELS)
        groups = labels_df.iloc[valid_idx]["subject_id"].values
        if len(y) < 50 or len(set(groups)) < args.n_splits:
            print(f"  {label_name:<16} n={len(y)}  (skipped)")
            continue
        results.append(probe_one_label(
            X=embeddings[valid_idx], y=y, groups=groups,
            label_name=label_name, is_clf=lcfg["type"] == "classification",
            n_splits=args.n_splits, use_groups=True,
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
                "seed": RANDOM_STATE,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
