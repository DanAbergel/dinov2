"""Fine-tuning (SFT) of a pretrained fMRI encoder — Brain-JEPA's headline protocol.

Point 4 of the v2 plan: instead of a frozen-encoder probe, we UNFREEZE the encoder
and train it end-to-end on the downstream task (+ a linear head on the CLS token).
This is the strongest downstream protocol and the one Brain-JEPA reports as its
headline (it also reports linear probing, which is our v2 probe).

Leakage-free, same 70:30 as the probe so the adaptation ladder is apples-to-apples
on the SAME held-out test set:
  - TRAIN  = 70% subjects (split=='train'); we carve a subject-level 15% val' from
             it for early stopping / model selection.
  - TEST   = the held-out 30% (val+test, excluded from pretraining), touched once.
Per scan we sample a RANDOM temporal window (train) / average sliding windows (eval),
resampled to T_FIXED @ 0.72 s — identical windowing to the probe.

Unfreeze depth (the point-4 axis):
  - head   : encoder frozen (= linear probe baseline)
  - last3  : unfreeze the last 3 transformer blocks + final norm (+ head)
  - all    : unfreeze the whole encoder (+ head)

Metrics reported on TEST: AUC / Accuracy / F1 (binary), mean±std over --seeds runs
(Brain-JEPA averages 5). Writes finetune_<ds>_<depth>.json into the run dir.

Usage (SLURM, needs GPU):
    python tasks/v2/finetune/finetune.py --run-dir .../runs/v1/base --dataset ADNI --depth last3
    SMOKE fast check: add --smoke
"""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dinov2.data.fmri_data import (
    _load_mmap, _temporal_resample, _zscore_per_frame, TARGET_TR,
)

# reuse the sibling probe.py machinery (encoder loader + table builders)
_PROBE = Path(__file__).resolve().parents[1] / "probe" / "probe.py"
_spec = importlib.util.spec_from_file_location("v2probe", _PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_FIXED = probe.T_FIXED
BUILDERS = {"ADNI": probe.build_table_adni,
            "ABIDE": probe.build_table_abide,
            "HCP": probe.build_table_hcp}


# ---------------- data ----------------

def _window(scan, tr, train):
    """(T,X,Y,Z) mmap -> one (T_FIXED,1,X,Y,Z) clip. Random window if train,
    else the centre window (deterministic)."""
    win = max(1, round(T_FIXED * TARGET_TR / tr))
    scan = scan.float()
    if scan.ndim == 4:
        scan = scan.unsqueeze(1)                       # (T,1,X,Y,Z)
    T = scan.shape[0]
    hi = max(T - win, 0)
    s = random.randint(0, hi) if train else hi // 2
    clip = scan[s:s + win].clone()
    clip = _temporal_resample(clip, T_FIXED)
    return _zscore_per_frame(clip)                     # (T_FIXED,1,X,Y,Z)


class ScanDS(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train          # items: [(path, tr, y)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, tr, y = self.items[i]
        clip = _window(_load_mmap(path), tr, self.train)
        return clip, torch.tensor(float(y))


# ---------------- model ----------------

class FTModel(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):                               # x: (B,T_FIXED,1,X,Y,Z)
        cls = self.encoder(x, is_training=True)["x_norm_clstoken"]
        return self.head(cls)


def set_trainable(encoder, depth):
    for p in encoder.parameters():
        p.requires_grad = (depth == "all")
    if depth == "last3":
        mods = list(encoder.blocks[-3:]) + [encoder.norm]
        for m in mods:
            for p in m.parameters():
                p.requires_grad = True
    n = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  encoder trainable params ({depth}): {n:,}", flush=True)


# ---------------- train / eval ----------------

@torch.no_grad()
def eval_scans(model, items, windows=3):
    """Per scan: average softmax over `windows` sliding windows -> prob of class 1."""
    model.eval()
    ys, ps = [], []
    for path, tr, y in items:
        scan = _load_mmap(path).float()
        if scan.ndim == 4:
            scan = scan.unsqueeze(1)
        T, win = scan.shape[0], max(1, round(T_FIXED * TARGET_TR / tr))
        starts = np.linspace(0, max(T - win, 0), windows).astype(int)
        probs = []
        for s in starts:
            clip = _zscore_per_frame(_temporal_resample(scan[s:s + win].clone(), T_FIXED))
            logits = model(clip.unsqueeze(0).to(DEVICE))
            probs.append(torch.softmax(logits, -1)[0, 1].item())
        ys.append(y)
        ps.append(float(np.mean(probs)))
    return np.array(ys), np.array(ps)


def run_once(encoder_builder, items_tr, items_va, items_te, depth, seed,
             epochs, batch_size, lr_enc, lr_head, embed_dim):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = FTModel(encoder_builder(), embed_dim).to(DEVICE)
    set_trainable(model.encoder, depth)

    ys_tr = np.array([y for _, _, y in items_tr])
    w = torch.tensor([1.0, (ys_tr == 0).sum() / max((ys_tr == 1).sum(), 1)],
                     dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    params = [{"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr_enc},
              {"params": model.head.parameters(), "lr": lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    loader = torch.utils.data.DataLoader(ScanDS(items_tr, True), batch_size=batch_size,
                                         shuffle=True, num_workers=4, drop_last=False)

    best_auc, best_state, patience, bad = -1.0, None, 5, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.long().to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        yv, pv = eval_scans(model, items_va)
        auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else 0.5
        print(f"    [seed {seed}] epoch {ep+1}/{epochs}  val_auc={auc:.3f}", flush=True)
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    yt, pt = eval_scans(model, items_te)
    pred = (pt >= 0.5).astype(int)
    return {"test_auc": float(roc_auc_score(yt, pt)) if len(set(yt)) > 1 else 0.5,
            "test_acc": float(accuracy_score(yt, pred)),
            "test_f1": float(f1_score(yt, pred, zero_division=0)),
            "val_auc": float(best_auc)}


def finetune_label(table, col, run_dir, ckpt, depth, seeds, epochs, batch_size,
                   lr_enc, lr_head):
    subj = np.array([t["subject"] for t in table])
    split = np.array([t["split"] for t in table])
    y = np.array([probe._to_float(t["row"].get(col)) for t in table], dtype=float)
    ok = ~np.isnan(y)
    tr_mask = (split == "train") & ok
    te_mask = ((split == "val") | (split == "test")) & ok
    if tr_mask.sum() < 10 or te_mask.sum() < 10 or len(np.unique(y[tr_mask])) < 2:
        return None

    items_all = [(t["path"], t["tr"], y[i]) for i, t in enumerate(table)]
    te_items = [items_all[i] for i in np.where(te_mask)[0]]

    # subject-level 85/15 carve of the train pool for early stopping
    tr_subj = sorted(set(subj[tr_mask]))
    random.Random(0).shuffle(tr_subj)
    n_val = max(1, int(0.15 * len(tr_subj)))
    va_subj = set(tr_subj[:n_val])
    tr_items = [items_all[i] for i in np.where(tr_mask)[0] if subj[i] not in va_subj]
    va_items = [items_all[i] for i in np.where(tr_mask)[0] if subj[i] in va_subj]

    def builder():
        enc, _ = probe.load_teacher(run_dir, ckpt)
        return enc

    embed_dim = probe.load_teacher(run_dir, ckpt)[1]
    results = []
    for s in seeds:
        results.append(run_once(builder, tr_items, va_items, te_items, depth, s,
                                epochs, batch_size, lr_enc, lr_head, embed_dim))
    agg = {k: (float(np.mean([r[k] for r in results])),
               float(np.std([r[k] for r in results]))) for k in results[0]}
    return {"mean_std": agg, "runs": results, "depth": depth,
            "n_train": len(tr_items), "n_val": len(va_items), "n_test": len(te_items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="ADNI", choices=list(BUILDERS))
    ap.add_argument("--depth", default="last3", choices=["head", "last3", "all"])
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2,3,4 for 5 runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, tiny — pipeline check")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    epochs = args.epochs
    if args.smoke:
        seeds, epochs = [0], 1

    print(f"Device={DEVICE}  run={Path(args.run_dir).name}  dataset={args.dataset}  depth={args.depth}",
          flush=True)
    table, LABELS = BUILDERS[args.dataset]()
    print(f"{args.dataset}: {len(table)} scans with split+label", flush=True)

    results = {}
    for name, col in LABELS.items():
        print(f"\n==== fine-tune  {args.dataset} · {name}  (depth={args.depth}) ====", flush=True)
        t0 = time.time()
        r = finetune_label(table, col, args.run_dir, args.checkpoint, args.depth,
                           seeds, epochs, args.batch_size, args.lr_encoder, args.lr_head)
        results[name] = r
        if r:
            m = r["mean_std"]
            print(f"  {name}: AUC {m['test_auc'][0]:.3f}±{m['test_auc'][1]:.3f}  "
                  f"Acc {m['test_acc'][0]:.3f}±{m['test_acc'][1]:.3f}  "
                  f"F1 {m['test_f1'][0]:.3f}±{m['test_f1'][1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"  {name}: skipped (too few samples / one class)", flush=True)

    out = Path(args.run_dir) / f"finetune_{args.dataset.lower()}_{args.depth}.json"
    out.write_text(json.dumps({"run": Path(args.run_dir).name, "dataset": args.dataset,
                               "depth": args.depth, "seeds": seeds, "results": results}, indent=2))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
