"""Orchestration: samples -> embeddings (cached) -> 70:30 -> train head per task -> JSON.

This is the glue that ties the modules together. The encoder runs once per scan
(embeddings cached on disk); then every task of the dataset is probed with a fresh
head on the 70:30 split.
"""

import json
from pathlib import Path

import numpy as np

from .datasets import _f
from .encoder import cls_token
from .head import train_probe
from .splits import split_masks


def run(dataset, teacher, run_dir, checkpoint, mlp, out):
    S = dataset.samples()
    print(f"{dataset.name}: {len(S)} scans", flush=True)

    # extract (or load-cached) one embedding per scan
    cache = Path(run_dir) / f"emb_{dataset.name}_{checkpoint.replace('.', '_')}.npz"
    if cache.exists() and np.load(cache)["X"].shape[0] == len(S):
        X = np.load(cache)["X"]
    else:
        X = np.stack([cls_token(teacher, s["path"], s["tr"]) for s in S])
        np.savez(cache, X=X)

    subj = np.array([s["subject"] for s in S])
    tr_all, te_all = split_masks(dataset.name, subj)

    results = {}
    for task, col in dataset.tasks.items():
        y = np.array([_f(s["labels"].get(col)) for s in S], dtype=float)
        ok = ~np.isnan(y)                            # drop scans with no label for this task
        tr, te = tr_all & ok, te_all & ok
        if tr.sum() < 10 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            results[task] = None
            print(f"  {task:16} (skipped)", flush=True)
            continue
        r = train_probe(X[tr], y[tr], X[te], y[te], hidden=mlp or ())
        results[task] = r
        print(f"  {task:16} AUC {r['auc']:.3f}  Acc {r['acc']:.3f}  F1 {r['f1']:.3f}  n_te={r['n_test']}",
              flush=True)

    payload = {"run": Path(run_dir).name, "dataset": dataset.name, "mlp": mlp, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"Saved {out}", flush=True)
    return payload
