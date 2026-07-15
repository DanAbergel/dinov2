"""Linear probe on the Imagenette-trained DINOv2 backbone.

Loads the trained teacher backbone (from the run's FSDP checkpoint, via DINOv2's
own SSLMetaArch + FSDPCheckpointer so the format matches exactly), extracts frozen
CLS features on Imagenette train + val, fits a logistic regression, and reports
val top-1 accuracy. This is the REAL test of whether the model learned — the DINO
loss curve cannot tell us (it sits at an equilibrium set by temperature/centering).

Run (via probe.sh):
    python probe.py --config-file <cfg> --output-dir <run_dir> \
        --train-root <imagenette>/train --val-root <imagenette>/val
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.fsdp import FSDPCheckpointer


def eval_transform():
    # Standard SSL eval transform: resize 256 -> center-crop 224 -> ImageNet norm.
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
    parser = get_args_parser(add_help=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    args = parser.parse_args()

    cfg = setup(args)

    # Build the exact same model DINOv2 trained, FSDP-wrap it, and load the run's
    # checkpoint through its own checkpointer (guaranteed format match). resume=True
    # loads the latest checkpoint pointed to by `last_checkpoint` in output_dir.
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    FSDPCheckpointer(model, cfg.train.output_dir).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    backbone = model.teacher.backbone
    backbone.eval()

    tf = eval_transform()
    print("extracting train features...", flush=True)
    Xtr, ytr = extract(backbone, args.train_root, tf)
    print("extracting val features...", flush=True)
    Xva, yva = extract(backbone, args.val_root, tf)
    print(f"features: train {Xtr.shape}, val {Xva.shape}", flush=True)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xva))
    acc = accuracy_score(yva, pred)

    n_classes = len(np.unique(ytr))
    print("\n==================== LINEAR PROBE ====================")
    print(f"  classes           : {n_classes}")
    print(f"  chance accuracy   : {100.0 / n_classes:.2f}%")
    print(f"  val top-1 accuracy: {acc * 100:.2f}%")
    print("  => LEARNS" if acc > 2.0 / n_classes else "  => at/near chance (NOT learning)")
    print("======================================================")


if __name__ == "__main__":
    main()
