"""Linear probe on a brain-image DINOv2 backbone, with labels from a CSV + a
stratified train/test split (default 80:20).

Unlike the Imagenette probe (which used ImageFolder class subdirs as labels), the
brain PNGs live in one folder and their label comes from HCP_YA_subjects.csv:
the subject id is parsed from the PNG name (subject_<ID>_...png) and looked up in
the CSV `Subject` column -> `--label-col` (default Gender, i.e. sex, chance 50%).

Run (via probe_brain.sh):
  python probe_brain.py --config-file <cfg> --output-dir <brain_run_dir> \
      --features-root <brain2d> --labels-csv <HCP_YA_subjects.csv> \
      --label-col Gender --test-frac 0.2  [dino.head_n_prototypes=... ...]
"""
import argparse
import csv
import os
import re

import numpy as np
import torch

_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.fsdp import FSDPCheckpointer


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract(backbone, root, tf, bs=128, workers=8, avgpool=False):
    ds = datasets.ImageFolder(root, transform=tf)     # one folder; we ignore its label
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True)
    feats = []
    for x, _ in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = backbone.forward_features(x)
            f = out["x_norm_clstoken"]
            if avgpool:                                # DINOv2 official probe: CLS ++ mean(patch tokens)
                f = torch.cat([f, out["x_norm_patchtokens"].mean(dim=1)], dim=-1)
            feats.append(f.float().cpu().numpy())
    paths = [p for p, _ in ds.samples]
    return np.concatenate(feats), paths


def subject_id(path):
    m = re.search(r"subject_(\d+)", os.path.basename(path))
    return m.group(1) if m else None


def main():
    ap = get_args_parser(add_help=True)
    ap.add_argument("--features-root", required=True, help="ImageFolder root with the brain PNGs")
    ap.add_argument("--labels-csv", required=True, help="HCP_YA_subjects.csv")
    ap.add_argument("--label-col", default="Gender", help="CSV column to predict (Gender=sex)")
    ap.add_argument("--id-col", default="Subject")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=0,
                    help="if >1: k-fold GROUP cross-validation (by subject) -> mean±std, robust to split noise")
    ap.add_argument("--avgpool", action="store_true",
                    help="DINOv2 official probe representation: CLS token ++ mean(patch tokens) (captures iBOT signal)")
    args = ap.parse_args()

    cfg = setup(args)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    FSDPCheckpointer(model, cfg.train.output_dir).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    backbone = model.teacher.backbone
    backbone.eval()

    print(f"extracting brain features... (avgpool={args.avgpool})", flush=True)
    X, paths = extract(backbone, args.features_root, eval_transform(), avgpool=args.avgpool)

    # subject_id -> label
    labmap = {}
    for r in csv.DictReader(open(args.labels_csv)):
        labmap[str(r[args.id_col]).strip()] = str(r[args.label_col]).strip()
    classes = sorted({v for v in labmap.values() if v and v.lower() != "nan"})
    cls2i = {c: i for i, c in enumerate(classes)}

    keep, y, groups = [], [], []
    for i, p in enumerate(paths):
        sid = subject_id(p) or ""
        lab = labmap.get(sid)
        if lab in cls2i:
            keep.append(i)
            y.append(cls2i[lab])
            groups.append(sid)                 # subject id -> split by SUBJECT (no leakage)
    X, y, groups = X[keep], np.array(y), np.array(groups)
    n_subj = len(set(groups))
    print(f"matched {len(y)}/{len(paths)} images, {n_subj} subjects, classes={classes}", flush=True)
    if n_subj < 10 or len(classes) < 2:
        raise SystemExit("not enough labelled subjects / classes to probe")

    # k-fold GROUP cross-validation (by subject) -> robust mean±std, immune to split luck
    if args.cv > 1:
        accs = []
        for tr, te in GroupKFold(n_splits=args.cv).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
        accs = np.array(accs)
        print(f"\n=== {args.cv}-FOLD GROUP CV ({args.label_col}) ===")
        print(f"  CV accuracy: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%   (chance {100.0/len(classes):.1f}%)")
        print(f"  folds: {['%.1f' % (a*100) for a in accs]}")
        print("========================================")
        # fall through to also print one detailed split below

    # split by SUBJECT (all frames of a subject stay together) — avoids leakage with multi-frame
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=0).split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"split: {len(set(groups[tr]))} train subj / {len(set(groups[te]))} test subj  "
          f"({len(tr)}/{len(te)} images)", flush=True)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))

    acc = accuracy_score(yte, pred)
    print(f"\n============ BRAIN PROBE: {args.label_col} ============")
    print(f"  train/test        : {len(ytr)}/{len(yte)}  ({int((1-args.test_frac)*100)}:{int(args.test_frac*100)} stratified)")
    print(f"  classes           : {classes}   (chance = {100.0/len(classes):.2f}%)")
    print(f"  test top-1 accuracy: {acc*100:.2f}%")
    print(f"  balanced accuracy  : {balanced_accuracy_score(yte, pred)*100:.2f}%")
    print(f"  F1 macro           : {f1_score(yte, pred, average='macro')*100:.2f}%")
    print(classification_report(yte, pred, target_names=classes, digits=3))
    print("=======================================================")


if __name__ == "__main__":
    main()
