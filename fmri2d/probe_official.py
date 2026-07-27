"""OFFICIAL DINOv2 linear-probe protocol, reproduced bit-for-bit on our brain PNGs + sex labels.

This reuses the exact classes / hyperparameters from ``dinov2/eval/linear.py``:
  * representation: create_linear_input (CLS of last n blocks [+ avgpool patch tokens])
  * classifier grid: setup_linear_classifiers over n_last_blocks_list=[1, 4] x avgpool{F,T}
                     x learning_rates (the 13 official values), scale_lr applied identically
  * optimizer:  torch.optim.SGD(momentum=0.9, weight_decay=0)
  * scheduler:  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
  * loss:       CrossEntropyLoss summed over all classifiers
  * iterations: epochs * epoch_length = 10 * 1250 = 12500 (official defaults)

The ONLY deviations from linear.py are the three the task allows:
  1. our PNG images (ImageFolder-style tree under --features-root),
  2. our labels (sex / Gender, num_classes=2) parsed from a CSV,
  3. a SUBJECT-LEVEL train/val split (train subjects disjoint from val subjects; no leakage).

We reproduce the run_eval_linear / eval_linear training loop in single-GPU form (see NOTE below)
rather than calling run_eval_linear directly, because that function is hard-wired to
make_dataset(dataset_str) + distributed SHARDED_INFINITE samplers + fvcore Checkpointer +
torchmetrics MetricType, none of which fit a custom subject-level split. Every optimizer /
scheduler / loss / hyperparameter is kept identical.

The checkpoint is loaded exactly like fmri2d/probe_brain.py: build a NON-FSDP teacher backbone
via build_model_from_cfg(only_teacher=True) and copy the teacher's backbone weights into it
(get_intermediate_layers / ModelWithIntermediateLayers do NOT work on the FSDP-wrapped module).
"""
import os
import re
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- Torch 2.6 shim: force weights_only=False (copied from probe_brain.py) ---
_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.fsdp import FSDPCheckpointer
from dinov2.models import build_model_from_cfg

from dinov2.data.transforms import (
    make_classification_train_transform,
    make_classification_eval_transform,
)
from dinov2.eval.setup import get_autocast_dtype
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.eval.linear import setup_linear_classifiers  # exact official classifier grid

_SUBJ = re.compile(r"subject_(\d+)")


# ---------------------------------------------------------------------------
# Dataset: brain PNGs under --features-root, sex label from CSV, subject-level split
# ---------------------------------------------------------------------------
class BrainSexDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), int(self.labels[idx])


def _scan_pngs(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def _load_labelmap(labels_csv, id_col, label_col):
    import csv
    labmap = {}
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            labmap[str(r[id_col]).strip()] = str(r[label_col]).strip()
    return labmap


def build_subject_split(features_root, labels_csv, id_col, label_col, val_frac, seed):
    """Return (train_ds_paths, train_labels, val_paths, val_labels, classes) with a
    SUBJECT-LEVEL split: every subject is entirely in train xor val (no leakage)."""
    labmap = _load_labelmap(labels_csv, id_col, label_col)
    classes = sorted({v for v in labmap.values() if v and v.lower() != "nan"})
    cls2i = {c: i for i, c in enumerate(classes)}

    paths, subjects, ys = [], [], []
    for p in _scan_pngs(features_root):
        m = _SUBJ.search(os.path.basename(p))
        lab = labmap.get(m.group(1)) if m else None
        if lab in cls2i:
            paths.append(p)
            subjects.append(m.group(1))
            ys.append(cls2i[lab])

    # deterministic, seeded subject-level split
    uniq = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_frac))
    val_subj = set(uniq[:n_val])

    tr_p, tr_y, va_p, va_y = [], [], [], []
    for p, s, y in zip(paths, subjects, ys):
        if s in val_subj:
            va_p.append(p); va_y.append(y)
        else:
            tr_p.append(p); tr_y.append(y)

    # sanity: no subject appears in both splits
    tr_subj = {s for p, s, y in zip(paths, subjects, ys) if s not in val_subj}
    assert tr_subj.isdisjoint(val_subj), "subject leakage between train and val!"

    return tr_p, tr_y, va_p, va_y, classes


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Evaluation: plain top-1 accuracy per classifier (num_classes=2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_all(feature_model, linear_classifiers, val_loader, device):
    linear_classifiers.eval()
    names = list(linear_classifiers.classifiers_dict.keys())
    correct = {k: 0 for k in names}
    total = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = feature_model(data)
        outputs = linear_classifiers(features)  # {name: logits}
        for k, logits in outputs.items():
            correct[k] += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return {k: correct[k] / max(total, 1) for k in names}


def main():
    ap = get_args_parser(add_help=True)
    ap.add_argument("--features-root", required=True, help="ImageFolder root with the brain PNGs")
    ap.add_argument("--labels-csv", required=True, help="HCP_YA_subjects.csv")
    ap.add_argument("--label-col", default="Gender", help="CSV column to predict (Gender=sex)")
    ap.add_argument("--id-col", default="Subject")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of SUBJECTS held out for val")
    # official linear.py defaults: epochs=10, epoch_length=1250 -> max_iter=12500
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epoch-length", type=int, default=1250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-period-iterations", type=int, default=1250)
    # the exact 13 learning rates from linear.py get_args_parser defaults
    ap.add_argument(
        "--learning-rates", nargs="+", type=float,
        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # official run_eval_linear uses seed = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    # ---- load OUR checkpoint exactly like probe_brain.py -> NON-FSDP teacher backbone ----
    cfg = setup(args)
    model = SSLMetaArch(cfg).to(device)
    model.prepare_for_distributed_training()
    FSDPCheckpointer(model, cfg.train.output_dir).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    backbone, _ = build_model_from_cfg(cfg, only_teacher=True)
    backbone = backbone.cuda().eval()
    sd = {k[len("backbone."):]: v for k, v in model.teacher.state_dict().items() if k.startswith("backbone.")}
    backbone.load_state_dict(sd, strict=True)

    # ---- wrap with ModelWithIntermediateLayers exactly like run_eval_linear ----
    n_last_blocks_list = [1, 4]                       # linear.py:503
    n_last_blocks = max(n_last_blocks_list)           # linear.py:504
    autocast_dtype = get_autocast_dtype(cfg)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)  # linear.py:505
    feature_model = ModelWithIntermediateLayers(backbone, n_last_blocks, autocast_ctx)   # linear.py:506

    # ---- datasets: official train/eval transforms, subject-level split ----
    tr_p, tr_y, va_p, va_y, classes = build_subject_split(
        args.features_root, args.labels_csv, args.id_col, args.label_col, args.val_frac, args.seed,
    )
    num_classes = len(classes)
    print(f"classes={classes} num_classes={num_classes}", flush=True)
    print(f"train images={len(tr_p)} (subjects disjoint from val), val images={len(va_p)}", flush=True)
    if num_classes < 2 or len(tr_p) == 0 or len(va_p) == 0:
        raise SystemExit("need >=2 classes and non-empty train/val splits")

    train_dataset = BrainSexDataset(tr_p, tr_y, make_classification_train_transform())
    val_dataset = BrainSexDataset(va_p, va_y, make_classification_eval_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False, pin_memory=True,
    )

    # ---- classifiers: EXACT official grid (create_linear_input + scale_lr inside) ----
    sample = train_dataset[0][0].unsqueeze(0).cuda()          # linear.py:507
    sample_output = feature_model(sample)
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output, n_last_blocks_list, args.learning_rates, args.batch_size, num_classes,
    )                                                          # linear.py:509-515

    # ---- optimizer / scheduler / loop: reproduced faithfully from eval_linear ----
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)   # linear.py:517
    max_iter = args.epochs * args.epoch_length                                       # linear.py:518
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)  # linear.py:519

    print(f"training {len(linear_classifiers)} linear classifiers for {max_iter} iterations "
          f"(bs={args.batch_size}, lr scaled by bs/256)", flush=True)

    train_iter = _infinite(train_loader)
    for iteration in range(max_iter):
        data, labels = next(train_iter)
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)
        losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, labels) for k, v in outputs.items()}  # linear.py:355
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 100 == 0:
            print(f"iter {iteration}/{max_iter}  loss={loss.item():.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)

        # periodic eval (best classifier so far), matching eval_period semantics
        if args.eval_period_iterations > 0 and (iteration + 1) % args.eval_period_iterations == 0 \
                and iteration != max_iter - 1:
            accs = evaluate_all(feature_model, linear_classifiers, val_loader, device)
            best_k = max(accs, key=accs.get)
            print(f"  [iter {iteration}] best val top-1 = {accs[best_k]*100:.2f}% ({best_k})", flush=True)
            linear_classifiers.train()

    # ---- final evaluation over the whole grid ----
    accs = evaluate_all(feature_model, linear_classifiers, val_loader, device)
    best_k = max(accs, key=accs.get)

    print("\n==================== OFFICIAL DINOv2 LINEAR PROBE ====================")
    print(f"  label = {args.label_col} (num_classes={num_classes}), subject-level val split "
          f"(val_frac={args.val_frac})")
    print(f"  grid  = n_last_blocks {n_last_blocks_list} x avgpool[F,T] x {len(args.learning_rates)} LRs "
          f"= {len(linear_classifiers)} classifiers, {max_iter} iters")
    print(f"  BEST  = {best_k}")
    print(f"  BEST val top-1 accuracy = {accs[best_k]*100:.2f}%")
    print("---------------------------------------------------------------------")
    for k in sorted(accs, key=accs.get, reverse=True):
        print(f"    {accs[k]*100:6.2f}%   {k}")
    print("=====================================================================")


if __name__ == "__main__":
    main()
