"""Leakage-free linear probe (ADNI or ABIDE) for a trained fMRI run.

Loads the run's teacher encoder, extracts ONE embedding per scan (mean CLS over
sliding windows resampled from the scan's native TR to 0.72s), then for each
label fits LogReg on TRAIN subjects, selects C on VAL, evaluates on TEST.
Subjects come from subject_split.json — val/test were held out of pretraining
(no leakage).

Usage (SLURM job, needs GPU):
    python probe.py --run-dir .../runs/fmri_v2_baseline --dataset ADNI
    python probe.py --run-dir .../runs/fmri_v2_baseline --dataset ABIDE
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from dinov2.models import build_model_from_cfg
from dinov2.data.fmri_data import (
    _load_mmap, _temporal_resample, _zscore_per_frame,
    TARGET_TR, LAB_ROOT, DEFAULT_SPLIT, DEFAULT_MANIFEST,
)

LAB = Path(LAB_ROOT)
ADNI_DIR = LAB / "ADNI_data" / "downsampled"
ADNI_MANIFEST = ADNI_DIR / "adni_manifest.csv"
ABIDE_PHENO = LAB / "ABIDE_data" / "abide_phenotypic.csv"
CORPUS_MANIFEST = LAB / DEFAULT_MANIFEST
T_FIXED = 270
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C_GRID = [0.01, 0.1, 1.0, 10.0]

LABELS_ADNI = {
    "degradation_1y": "degradation_binary_1year",
    "degradation_2y": "degradation_binary_2years",
    "degradation_3y": "degradation_binary_3years",
    "CDR": "CDR_Binary",
    "Sex": "Sex_Binary",
}
LABELS_ABIDE = {"Autism": "autism", "Sex": "sex_bin"}   # built in build_table_abide


# ---------------- encoder ----------------

def load_teacher(run_dir, checkpoint):
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    _student, teacher, embed_dim = build_model_from_cfg(cfg)
    ckpt = torch.load(Path(run_dir) / checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    tb = {}
    for k, v in state.items():
        kk = k.replace("_fsdp_wrapped_module.", "")
        if kk.startswith("teacher.backbone."):
            tb[kk[len("teacher.backbone."):]] = v
    if not tb:
        raise RuntimeError(f"No teacher.backbone.* keys. Sample: {list(state)[:8]}")
    missing, unexpected = teacher.load_state_dict(tb, strict=False)
    print(f"teacher: loaded {len(tb)} keys  missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    return teacher.to(DEVICE).eval(), embed_dim


@torch.no_grad()
def scan_embedding(teacher, path, native_tr):
    """Mean CLS over sliding native windows (each spanning T_FIXED*0.72s),
    resampled to T_FIXED frames at 0.72s. Cropping native FIRST keeps
    resample_poly cheap (small up/down)."""
    win = max(1, round(T_FIXED * TARGET_TR / native_tr))
    stride = max(1, win // 2)
    scan = _load_mmap(path).float()                  # (T, X, Y, Z)
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                     # (T, 1, X, Y, Z)
    T = scan.shape[0]
    embs = []
    for s in range(0, max(T - win + 1, 1), stride):
        clip = scan[s:s + win].clone()
        clip = _temporal_resample(clip, T_FIXED)     # -> 270 @ 0.72s
        clip = _zscore_per_frame(clip)
        x = clip.unsqueeze(0).to(DEVICE)             # (1, T_FIXED, 1, X, Y, Z)
        out = teacher(x, is_training=True)
        embs.append(out["x_norm_clstoken"].squeeze(0).float().cpu())
    return torch.stack(embs).mean(0).numpy()


# ---------------- labels / split ----------------

def _to_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _split_map(ds):
    split = json.loads((LAB / DEFAULT_SPLIT).read_text())["datasets"][ds]
    return {s: name for name, subs in split.items() for s in subs}


def build_table_adni():
    sub2split = _split_map("ADNI")
    table = []
    for r in csv.DictReader(open(ADNI_MANIFEST)):
        sid, iid = r["subject_id"], r["image_id"]
        p = ADNI_DIR / sid / f"{iid}.pt"
        sp = sub2split.get(sid)
        if sp is None or not p.exists():
            continue
        table.append({"path": p, "subject": sid, "split": sp, "tr": 3.0, "row": r})
    return table, LABELS_ADNI


def build_table_abide():
    sub2split = _split_map("ABIDE")
    pheno = {r["FILE_ID"]: r for r in csv.DictReader(open(ABIDE_PHENO))}
    table = []
    for r in csv.DictReader(open(CORPUS_MANIFEST)):
        if r["dataset"] != "ABIDE":
            continue
        sid = r["subject_id"]
        sp = sub2split.get(sid)
        if sp is None:
            continue
        ph = pheno.get(sid.replace("_downsampled", ""))   # FILE_ID = name w/o suffix
        if ph is None:
            continue
        dx = _to_float(ph.get("DX_GROUP"))                # 1=autism, 2=control
        if dx is None:
            continue
        sex = _to_float(ph.get("SEX"))                    # 1=male, 2=female
        row = {"autism": 1.0 if dx == 1 else 0.0,
               "sex_bin": (1.0 if sex == 1 else 0.0) if sex is not None else float("nan")}
        table.append({"path": Path(r["path"]), "subject": sid, "split": sp,
                      "tr": float(r["tr"]), "row": row})
    return table, LABELS_ABIDE


# ---------------- probe ----------------

def probe_label(X, y, splits, name):
    tr, va, te = (splits == "train"), (splits == "val"), (splits == "test")
    m = ~np.isnan(y)
    tr, va, te = tr & m, va & m, te & m
    if tr.sum() < 10 or te.sum() < 5 or va.sum() < 5:
        return None
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2 or len(np.unique(y[va])) < 2:
        return None

    sc = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])

    best_c, best_val = None, -1
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
        if auc > best_val:
            best_val, best_c = auc, C

    clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
    test_auc = roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])
    return {"C": best_c, "val_auc": float(best_val), "test_auc": float(test_auc),
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "pos_test": int(y[te].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="ADNI", choices=["ADNI", "ABIDE"])
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    args = ap.parse_args()

    print(f"Device: {DEVICE}  cuda_available={torch.cuda.is_available()}", flush=True)
    teacher, embed_dim = load_teacher(args.run_dir, args.checkpoint)
    print(f"embed_dim={embed_dim}", flush=True)

    table, LABELS = build_table_adni() if args.dataset == "ADNI" else build_table_abide()
    print(f"{args.dataset} scans with split+label: {len(table)}", flush=True)
    if not table:
        raise RuntimeError(f"No {args.dataset} scans matched (labels/split missing?)")

    t0 = time.time()
    embs = []
    for i, t in enumerate(table):
        embs.append(scan_embedding(teacher, t["path"], t["tr"]))
        if (i + 1) % 100 == 0 or i == len(table) - 1:
            dt = time.time() - t0
            print(f"  embeddings {i+1}/{len(table)}  ({dt:.0f}s, {dt/(i+1)*1000:.0f} ms/scan)",
                  flush=True)
    X = np.stack(embs)
    splits = np.array([t["split"] for t in table])

    run_name = Path(args.run_dir).name
    print(f"\n{'='*64}\n  PROBE  {run_name}  {args.dataset}  ({args.checkpoint})\n{'='*64}")
    print(f"  {'label':16} {'C':>5} {'val_auc':>8} {'TEST_AUC':>9}  {'n_tr/n_te':>10} pos_te")
    results = {}
    for name, col in LABELS.items():
        y = np.array([_to_float(t["row"].get(col)) for t in table], dtype=float)
        r = probe_label(X, y, splits, name)
        results[name] = r
        if r:
            print(f"  {name:16} {r['C']:>5} {r['val_auc']:>8.3f} {r['test_auc']:>9.3f}"
                  f"  {r['n_train']:>4}/{r['n_test']:<4} {r['pos_test']}", flush=True)
        else:
            print(f"  {name:16} (skipped — too few samples / one class)", flush=True)

    out = Path(args.run_dir) / f"probe_{args.dataset.lower()}.json"
    out.write_text(json.dumps({"run": run_name, "dataset": args.dataset,
                               "checkpoint": args.checkpoint, "results": results}, indent=2))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
