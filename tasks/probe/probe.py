"""Leakage-free linear probe on ADNI for a trained fMRI run.

Loads the teacher encoder from a run's checkpoint, extracts ONE embedding per
ADNI scan (mean CLS over sliding T_fixed windows resampled to TR=0.72s), then for
each label fits LogReg on TRAIN subjects, selects C on VAL, and evaluates on TEST.
Subjects come from subject_split.json — the val/test subjects were held out of
pretraining (no leakage).

Usage (SLURM job, needs GPU):
    python probe.py --run-dir /sci/.../runs/fmri_v2_baseline
                    [--checkpoint model_final.rank_0.pth]
"""

import argparse
import csv
import json
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
    TARGET_TR, LAB_ROOT, DEFAULT_SPLIT,
)

ADNI_DIR = Path(LAB_ROOT) / "ADNI_data" / "downsampled"
ADNI_MANIFEST = ADNI_DIR / "adni_manifest.csv"
ADNI_TR = 3.0
T_FIXED = 270
STRIDE = 135
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# label name -> manifest column (all binary classification)
LABELS = {
    "degradation_1y": "degradation_binary_1year",
    "degradation_2y": "degradation_binary_2years",
    "degradation_3y": "degradation_binary_3years",
    "CDR": "CDR_Binary",
    "Sex": "Sex_Binary",
}
C_GRID = [0.01, 0.1, 1.0, 10.0]


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
        raise RuntimeError(
            f"No 'teacher.backbone.*' keys found. Sample keys: {list(state)[:8]}"
        )
    missing, unexpected = teacher.load_state_dict(tb, strict=False)
    print(f"teacher: loaded {len(tb)} keys  missing={len(missing)} unexpected={len(unexpected)}")
    return teacher.to(DEVICE).eval(), embed_dim


# Native window (in ADNI frames) spanning T_FIXED * TARGET_TR seconds. We crop
# this small native window FIRST, then resample 65->270, matching training. The
# old "resample the whole 140->583 scan" made resample_poly use a ~5800-tap FIR
# over 109350 voxels -> minutes per scan. Cropping first keeps up/down small.
NATIVE_WIN = max(1, round(T_FIXED * TARGET_TR / ADNI_TR))     # ~65
NATIVE_STRIDE = max(1, NATIVE_WIN // 2)


@torch.no_grad()
def scan_embedding(teacher, path):
    scan = _load_mmap(path).float()                  # (T, X, Y, Z)
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                     # (T, 1, X, Y, Z)
    T = scan.shape[0]
    embs = []
    for s in range(0, max(T - NATIVE_WIN + 1, 1), NATIVE_STRIDE):
        clip = scan[s:s + NATIVE_WIN].clone()        # small native window
        clip = _temporal_resample(clip, T_FIXED)     # 65 -> 270 @ 0.72s (cheap)
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


def build_table():
    """Return list of dicts: {path, subject, split, labels...} for ADNI scans
    whose subject is in subject_split.json."""
    split = json.loads((Path(LAB_ROOT) / DEFAULT_SPLIT).read_text())["datasets"]["ADNI"]
    sub2split = {s: name for name, subs in split.items() for s in subs}
    rows = list(csv.DictReader(open(ADNI_MANIFEST)))
    table = []
    for r in rows:
        sid, iid = r["subject_id"], r["image_id"]
        p = ADNI_DIR / sid / f"{iid}.pt"
        sp = sub2split.get(sid)
        if sp is None or not p.exists():
            continue
        table.append({"path": p, "subject": sid, "split": sp, "row": r})
    return table


# ---------------- probe ----------------

def probe_label(X, y, splits, name):
    """Fit LogReg on train, pick C on val (AUC), report test AUC."""
    tr, va, te = (splits == "train"), (splits == "val"), (splits == "test")
    m = ~np.isnan(y)
    tr, va, te = tr & m, va & m, te & m
    if tr.sum() < 10 or te.sum() < 5 or va.sum() < 5:
        return None
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None

    sc = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])

    best_c, best_val = None, -1
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        if len(np.unique(y[va])) < 2:
            continue
        auc = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
        if auc > best_val:
            best_val, best_c = auc, C

    clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
    test_auc = roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])
    return {"C": best_c, "val_auc": best_val, "test_auc": test_auc,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "pos_test": int(y[te].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    args = ap.parse_args()

    print(f"Device: {DEVICE}  cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"native_win={NATIVE_WIN} stride={NATIVE_STRIDE} -> T_FIXED={T_FIXED}", flush=True)
    teacher, embed_dim = load_teacher(args.run_dir, args.checkpoint)
    print(f"embed_dim={embed_dim}", flush=True)

    table = build_table()
    print(f"ADNI scans with split+file: {len(table)}", flush=True)

    import time
    t0 = time.time()
    embs = []
    for i, t in enumerate(table):
        embs.append(scan_embedding(teacher, t["path"]))
        if (i + 1) % 50 == 0 or i == len(table) - 1:
            dt = time.time() - t0
            print(f"  embeddings {i+1}/{len(table)}  ({dt:.0f}s, {dt/(i+1)*1000:.0f} ms/scan)",
                  flush=True)
    X = np.stack(embs)
    splits = np.array([t["split"] for t in table])
    print(f"embeddings: {X.shape}", flush=True)

    run_name = Path(args.run_dir).name
    print(f"\n{'='*64}\n  PROBE  {run_name}  ({args.checkpoint})\n{'='*64}")
    print(f"  {'label':16} {'C':>5} {'val_auc':>8} {'TEST_AUC':>9}  {'n_tr/n_te':>10} pos_te")
    results = {}
    for name, col in LABELS.items():
        y = np.array([_to_float(t["row"].get(col)) for t in table], dtype=float)
        r = probe_label(X, y, splits, name)
        results[name] = r
        if r:
            print(f"  {name:16} {r['C']:>5} {r['val_auc']:>8.3f} {r['test_auc']:>9.3f}"
                  f"  {r['n_train']:>4}/{r['n_test']:<4} {r['pos_test']}")
        else:
            print(f"  {name:16} (skipped — too few samples / one class)")

    out = Path(args.run_dir) / "probe_adni.json"
    out.write_text(json.dumps({"run": run_name, "checkpoint": args.checkpoint,
                               "results": results}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
