"""Linear probe on the OFFICIAL pretrained DINOv2 ViT-S/14 (LVD-142M weights).

Same protocol as probe.py (frozen CLS features -> logistic regression -> Imagenette
val metrics), but the backbone is the PUBLISHED DINOv2 model built from our own repo
code (dinov2.hub.backbones.dinov2_vits14, pretrained=True) — no torch.hub clone, no
retraining. This is the "real DINO" reference: it shows the top of the spectrum
(≈95% on Imagenette) to contrast with our tiny from-scratch run (~41%).

Run (via probe_pretrained.sh):
    python probe_pretrained.py --train-root <imagenette>/train --val-root <imagenette>/val
"""
import argparse
import os

import numpy as np
import torch

# torch 2.6 defaults weights_only=True; force False for the trusted pretrained ckpt.
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

from dinov2.hub.backbones import dinov2_vits14


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def extract(backbone, root, tf, bs=128, workers=8):
    ds = datasets.ImageFolder(root, transform=tf)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True)
    feats, labels = [], []
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = backbone.forward_features(x)["x_norm_clstoken"]
        feats.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    args = ap.parse_args()

    print("building official pretrained DINOv2 ViT-S/14 (LVD-142M)...", flush=True)
    backbone = dinov2_vits14(pretrained=True).cuda().eval()

    tf = eval_transform()
    print("extracting train features...", flush=True)
    Xtr, ytr = extract(backbone, args.train_root, tf)
    print("extracting val features...", flush=True)
    Xva, yva = extract(backbone, args.val_root, tf)
    print(f"features: train {Xtr.shape}, val {Xva.shape}", flush=True)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xva))

    acc = accuracy_score(yva, pred)
    bal_acc = balanced_accuracy_score(yva, pred)
    f1_macro = f1_score(yva, pred, average="macro")
    f1_weighted = f1_score(yva, pred, average="weighted")
    n_classes = len(np.unique(ytr))

    wnid_names = {
        "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
        "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
        "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
        "n03888257": "parachute",
    }
    val_classes = sorted(os.listdir(args.val_root))
    target_names = [wnid_names.get(w, w) for w in val_classes]

    print("\n============ PROBE: OFFICIAL PRETRAINED DINOv2 ViT-S/14 ============")
    print(f"  classes            : {n_classes}   (chance = {100.0 / n_classes:.2f}%)")
    print(f"  val top-1 accuracy : {acc * 100:.2f}%")
    print(f"  balanced accuracy  : {bal_acc * 100:.2f}%")
    print(f"  F1 macro           : {f1_macro * 100:.2f}%")
    print(f"  F1 weighted        : {f1_weighted * 100:.2f}%")
    print("\n  per-class precision / recall / F1:")
    print(classification_report(yva, pred, target_names=target_names, digits=3))
    print("===================================================================")


if __name__ == "__main__":
    main()
