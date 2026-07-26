"""Online linear probe: given a LIVE backbone (during training), extract CLS(+patch) features
for a folder of labelled PNGs and return the subject-level group-CV accuracy (mean, std).

Used by train.py's periodic-probe hook (env-gated). Mirrors fmri2d/probe_brain.py's
representation (CLS token, or CLS ++ mean(patch tokens) when avgpool=True) and its
subject-level GroupKFold split (no leakage). Returns fast numbers to log an evolution curve.
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
def _extract(backbone, root, avgpool, bs=128, workers=8):
    ds = datasets.ImageFolder(root, transform=_eval_tf())
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True)
    feats = []
    for x, _ in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = backbone.forward_features(x)
            f = out["x_norm_clstoken"]
            if avgpool:
                f = torch.cat([f, out["x_norm_patchtokens"].mean(dim=1)], dim=-1)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats), [p for p, _ in ds.samples]


def probe_backbone(backbone, root, labels_csv, label_col="Gender", cv=5, avgpool=True, id_col="Subject"):
    """Return (mean_cv_acc, std) of a subject-level group-CV linear probe. Puts backbone back
    in its prior train/eval mode afterwards, so it's safe to call mid-training."""
    was_training = backbone.training
    backbone.eval()
    try:
        X, paths = _extract(backbone, root, avgpool)
    finally:
        if was_training:
            backbone.train()

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
    X, y, groups = X[keep], np.array(y), np.array(groups)
    if len(set(groups)) < cv or len(classes) < 2:
        return float("nan"), float("nan")

    accs = []
    for tr, te in GroupKFold(n_splits=cv).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
    accs = np.array(accs)
    return float(accs.mean()), float(accs.std())
