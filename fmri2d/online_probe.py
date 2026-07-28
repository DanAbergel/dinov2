"""Online linear probe on a LIVE backbone, using the OFFICIAL DINOv2 representation:
the CLS token of the last N blocks, optionally concatenated with the avg-pooled patch tokens
of the last block (create_linear_input). We sweep the official grid n_last_blocks in {1, N} x
use_avgpool in {False, True} and report the BEST config (like setup_linear_classifiers), with a
subject-level GroupKFold split (no leakage). Used by train.py's periodic-probe hook.
"""
import csv
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

_SUBJ = re.compile(r"subject_(\d+)")


def _eval_tf():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def _extract(backbone, root, n_blocks=4, bs=128, workers=8):
    """Return (cls_per_block: list of N arrays (imgs, D), patch_mean_last: (imgs, D), paths)."""
    ds = datasets.ImageFolder(root, transform=_eval_tf())
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True)
    cls_per_block = [[] for _ in range(n_blocks)]
    patch_mean = []
    for x, _ in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outs = backbone.get_intermediate_layers(x, n=n_blocks, return_class_token=True, norm=True)
        for b in range(n_blocks):
            cls_per_block[b].append(outs[b][1].float().cpu().numpy())
        patch_mean.append(outs[-1][0].mean(dim=1).float().cpu().numpy())
    cls_per_block = [np.concatenate(c) for c in cls_per_block]
    patch_mean = np.concatenate(patch_mean)
    return cls_per_block, patch_mean, ds.samples          # samples = list of (path, class_idx)


def _linear_input(cls_per_block, patch_mean, use_n_blocks, use_avgpool):
    out = np.concatenate(cls_per_block[-use_n_blocks:], axis=1)   # concat CLS of last n blocks
    if use_avgpool:
        out = np.concatenate([out, patch_mean], axis=1)          # + avgpool patch tokens (last block)
    return out


def probe_backbone(backbone, root, labels_csv, label_col="Gender", cv=5, avgpool=True, id_col="Subject", n_blocks=4):
    """Official-style probe: sweep n_last_blocks in {1, n_blocks} x avgpool in {F,T}, report best
    subject-level GroupKFold CV (mean, std). `avgpool` arg kept for signature compat (ignored: swept)."""
    was_training = backbone.training
    backbone.eval()
    try:
        cls_per_block, patch_mean, samples = _extract(backbone, root, n_blocks=n_blocks)
    finally:
        if was_training:
            backbone.train()

    paths = [p for p, _ in samples]
    if labels_csv == "IMAGEFOLDER":
        # Imagenette-style: label = ImageFolder class index, image-level StratifiedKFold (no subjects)
        y = np.array([t for _, t in samples])
        keep = np.arange(len(y))
        groups = None
        n_classes = len(set(y.tolist()))
    else:
        labmap = {}
        for r in csv.DictReader(open(labels_csv)):
            labmap[str(r[id_col]).strip()] = str(r[label_col]).strip()
        classes = sorted({v for v in labmap.values() if v and v.lower() != "nan"})
        cls2i = {c: i for i, c in enumerate(classes)}
        keep, y, groups = [], [], []
        for i, p in enumerate(paths):
            m = _SUBJ.search(os.path.basename(p))
            lab = labmap.get(m.group(1)) if m else None
            if lab in cls2i:
                keep.append(i)
                y.append(cls2i[lab])
                groups.append(m.group(1))
        keep = np.array(keep)
        y, groups = np.array(y), np.array(groups)
        n_classes = len(classes)
    if n_classes < 2 or len(y) < cv or (groups is not None and len(set(groups)) < cv):
        return float("nan"), float("nan")

    # subject-level GroupKFold for brains; image-level StratifiedKFold for Imagenette
    if groups is None:
        from sklearn.model_selection import StratifiedKFold
        folds = list(StratifiedKFold(n_splits=cv, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    else:
        folds = list(GroupKFold(n_splits=cv).split(np.zeros(len(y)), y, groups))

    best_mean, best_std, best_cfg = -1.0, 0.0, None
    for n in sorted({1, n_blocks}):
        for avg in (False, True):
            X = _linear_input(cls_per_block, patch_mean, n, avg)[keep]
            accs = []
            for tr, te in folds:
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
            m, s = float(np.mean(accs)), float(np.std(accs))
            if m > best_mean:
                best_mean, best_std, best_cfg = m, s, (n, avg)
    print(f"  [probe] best config: n_blocks={best_cfg[0]} avgpool={best_cfg[1]}", flush=True)
    return best_mean, best_std
