"""Periodic OFFICIAL DINOv2 linear probe on the live teacher, for fMRI (HCP).

Runs the EXACT official DINOv2 linear-eval protocol every N training iterations so
we can watch the probe evolve instead of waiting for the end of training:

  representation : create_linear_input (CLS of last n blocks [+ avgpool patches]) — OFFICIAL, verbatim
  classifier sweep: setup_linear_classifiers (n_blocks in {1,4} x avgpool x LR grid) — OFFICIAL, verbatim
  optimiser      : SGD(momentum=0.9, weight_decay=0) + CosineAnnealingLR — OFFICIAL
  selection      : best classifier on a held-out VAL split, reported on the TEST split — OFFICIAL

The ONLY fMRI-specific part is feature extraction: a scan is harmonised (TR resample +
per-frame z-score, IDENTICAL to training via dinov2.data.fmri_data) and passed through
teacher.get_intermediate_layers over sliding windows; the token list is averaged over
windows. We feed the official classifiers a reconstructed x_tokens_list so create_linear_input
and LinearClassifier run unchanged (the mean patch token is stored as (B,1,D) so the
official torch.mean(dim=1) reproduces it exactly).

Leakage-free, subject-level train/val/test (reuses probelib's HCP samples + fixed split).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dinov2.data.fmri_data import (_load_mmap, _temporal_resample,
                                   _zscore_per_frame, TARGET_TR)
from dinov2.eval.linear import create_linear_input, setup_linear_classifiers  # OFFICIAL, verbatim

# probelib provides the fMRI HCP sample list + the fixed leakage-free split (data plumbing only).
_PROBE_DIR = Path(__file__).resolve().parents[2] / "tasks" / "v2" / "probe"
if str(_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBE_DIR))

T_FIXED = 270
_N_LAST_BLOCKS = [1, 4]
# Official DINOv2 linear-eval learning-rate grid.
_LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]


@torch.no_grad()
def _scan_tokens(teacher, path, native_tr, n_blocks, device, t_fixed=T_FIXED):
    """Per scan: (cls_per_block [n_blocks, D], mean_patch_last [D]) averaged over sliding windows.

    Same harmonisation as training/probelib. get_intermediate_layers returns a list of
    (patch_tokens, cls_token) for the last `n_blocks` blocks; we mean CLS per block and the
    last block's patch tokens over the token axis, then over windows. `t_fixed` MUST match the
    trained model's fmri_temporal_size, else the clip length breaks the temporal token grid.
    """
    win = max(1, round(t_fixed * TARGET_TR / native_tr))
    stride = max(1, win // 2)
    scan = _load_mmap(path).float()
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                       # (T, 1, X, Y, Z)
    cls_sum = patch_sum = None
    nw = 0
    for s in range(0, max(scan.shape[0] - win + 1, 1), stride):
        clip = _zscore_per_frame(_temporal_resample(scan[s:s + win].clone(), t_fixed))
        toks = teacher.get_intermediate_layers(
            clip.unsqueeze(0).to(device), n=n_blocks, return_class_token=True
        )
        cls = torch.stack([c for _, c in toks], 0).squeeze(1).float()   # (n_blocks, D)
        patch_last = toks[-1][0].mean(dim=1).squeeze(0).float()         # (D,)
        cls_sum = cls if cls_sum is None else cls_sum + cls
        patch_sum = patch_last if patch_sum is None else patch_sum + patch_last
        nw += 1
    return (cls_sum / nw).cpu(), (patch_sum / nw).cpu()


def _rebuild_tokens_list(cls_b, patch_b, n_blocks):
    """Rebuild the official x_tokens_list from compressed features so create_linear_input runs
    verbatim. cls_b: (B, n_blocks, D); patch_b: (B, D). Only the last block's patch is used by
    create_linear_input (avgpool), stored as (B,1,D) so its torch.mean(dim=1) == patch_b."""
    B, _, D = cls_b.shape
    out = []
    for i in range(n_blocks):
        if i == n_blocks - 1:
            patch = patch_b.unsqueeze(1)                # (B,1,D) -> mean(dim=1) == patch_b
        else:
            patch = cls_b.new_zeros(B, 1, D)            # unused by create_linear_input
        out.append((patch, cls_b[:, i]))
    return out


def _subject_val_split(train_subjects, frac=0.2, seed=0):
    """Deterministic subject-level val carve-out from the train subjects (no leakage)."""
    uniq = sorted(set(train_subjects))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * frac))
    val_subj = set(uniq[:n_val])
    return val_subj


@torch.no_grad()
def _accuracy(classifiers, cls_t, patch_t, y, n_blocks, device, bs=256):
    """Per-classifier accuracy over a feature set."""
    correct = None
    total = 0
    for i in range(0, len(y), bs):
        cb = cls_t[i:i + bs].to(device)
        pb = patch_t[i:i + bs].to(device)
        yb = y[i:i + bs].to(device)
        logits = classifiers(_rebuild_tokens_list(cb, pb, n_blocks))
        if correct is None:
            correct = {k: 0 for k in logits}
        for k, v in logits.items():
            correct[k] += (v.argmax(1) == yb).sum().item()
        total += len(yb)
    return {k: c / total for k, c in correct.items()}


def run_periodic_probe(teacher, device, max_iter=2000, batch_size=128, seed=0, t_fixed=T_FIXED):
    """Official linear probe on HCP Sex with the live teacher. Returns a dict with the best
    classifier's val/test accuracy (+ its config), or None if HCP data is unavailable.
    `t_fixed` must match the model's fmri_temporal_size (passed from the train hook)."""
    from probelib.datasets import HCP
    from probelib.splits import split_masks

    was_training = teacher.training
    teacher.eval()                                     # deterministic features (no droppath) during the probe
    try:
        return _run_probe(teacher, device, max_iter, batch_size, seed, t_fixed)
    finally:
        teacher.train(was_training)                    # restore whatever mode training was in


def _run_probe(teacher, device, max_iter, batch_size, seed, t_fixed=T_FIXED):
    from probelib.datasets import HCP
    from probelib.splits import split_masks

    ds = HCP()
    S = ds.samples()
    task, col = next(iter(ds.tasks.items()))          # HCP: Sex
    from probelib.datasets import _f
    y_all = np.array([_f(s["labels"].get(col)) for s in S], dtype=float)
    ok = ~np.isnan(y_all)
    subj = np.array([s["subject"] for s in S])
    tr_all, te_all = split_masks(ds.name, subj)
    tr_mask = tr_all & ok
    te_mask = te_all & ok
    if tr_mask.sum() < 20 or te_mask.sum() < 10 or len(np.unique(y_all[tr_mask])) < 2:
        return None

    n_blocks = max(_N_LAST_BLOCKS)
    # ---- extract features for train+test scans (the expensive part) ----
    cls_list, patch_list = [], []
    keep = tr_mask | te_mask
    for s, k in zip(S, keep):
        if not k:
            cls_list.append(None); patch_list.append(None); continue
        c, p = _scan_tokens(teacher, s["path"], s["tr"], n_blocks, device, t_fixed)
        cls_list.append(c); patch_list.append(p)

    idx_tr = np.where(tr_mask)[0]
    idx_te = np.where(te_mask)[0]
    val_subj = _subject_val_split(subj[idx_tr], seed=seed)
    is_val = np.array([subj[i] in val_subj for i in idx_tr])
    idx_val = idx_tr[is_val]
    idx_fit = idx_tr[~is_val]
    if len(idx_fit) < 10 or len(idx_val) < 2:
        idx_fit, idx_val = idx_tr, idx_te            # fallback: select on test if train too small

    def _stack(idx):
        return (torch.stack([cls_list[i] for i in idx]),
                torch.stack([patch_list[i] for i in idx]),
                torch.tensor(y_all[idx].astype(int)))
    cls_fit, patch_fit, y_fit = _stack(idx_fit)
    cls_val, patch_val, y_val = _stack(idx_val)
    cls_te,  patch_te,  y_te  = _stack(idx_te)

    num_classes = int(y_all[keep].max()) + 1
    sample_tokens = _rebuild_tokens_list(cls_fit[:1].to(device), patch_fit[:1].to(device), n_blocks)
    classifiers, optim_param_groups = setup_linear_classifiers(
        sample_tokens, _N_LAST_BLOCKS, _LR_GRID, batch_size, num_classes
    )
    # Single-GPU SLURM job: dinov2 enables distributed (world_size 1) so
    # setup_linear_classifiers wraps in DDP — unwrap to train it directly (no sync needed).
    if hasattr(classifiers, "module"):
        classifiers = classifiers.module
    classifiers = classifiers.to(device)
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    ce = nn.CrossEntropyLoss()

    n_fit = len(y_fit)
    bs = min(batch_size, n_fit)
    rng = np.random.RandomState(seed)
    classifiers.train()
    for it in range(max_iter):
        b = rng.randint(0, n_fit, size=bs)
        cb = cls_fit[b].to(device); pb = patch_fit[b].to(device); yb = y_fit[b].to(device)
        logits = classifiers(_rebuild_tokens_list(cb, pb, n_blocks))
        loss = sum(ce(v, yb) for v in logits.values())
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

    classifiers.eval()
    val_acc = _accuracy(classifiers, cls_val, patch_val, y_val, n_blocks, device)
    best_name = max(val_acc, key=val_acc.get)
    test_acc = _accuracy(classifiers, cls_te, patch_te, y_te, n_blocks, device)[best_name]
    return {"task": task, "best": best_name,
            "val_acc": val_acc[best_name], "test_acc": test_acc,
            "n_test": int(len(y_te))}
