"""Aggregate probe_*.json across the v1 runs into a SOTA-comparison table.

One row per (axis, metric) where the metric is THE ONE THE SOTA REPORTS for that
task (AUROC for ABIDE-Autism/BNT, F1 for LCM, Acc for ADNI/Brain-JEPA, ...), so
every comparison is apples-to-apples. Columns = our runs (base, fourier).

A second table shows the point-3 ablation: linear probe vs MLP head (run=base),
if the *_mlp.json files exist.

Usage:
    python tasks/v2/probe/summarize.py
"""

import json
import os

LAB = "/sci/labs/arieljaffe/dan.abergel1"
RUNS = ["base", "fourier", "noblock2", "pool"]   # Phase-A pretraining ablation
# base/fourier are the frozen v1 baseline (runs/v1); noblock2/pool are the v2
# architecture ablation (runs/v2). Map each run to where its checkpoints live.
RUN_VERSION = {"base": "v1", "fourier": "v1", "noblock2": "v2", "pool": "v2"}

# (dataset, probe-label, metric key in json, axis label, metric name, SOTA ref)
AXES = [
    ("ABIDE", "Autism",    "test_auc", "ABIDE · Autism", "AUROC", "BNT 0.80 / BrainGFM 0.71"),
    ("ABIDE", "Autism",    "test_f1",  "ABIDE · Autism", "F1",    "LCM 0.73"),
    ("ABIDE", "Age",       "test_acc", "ABIDE · Age",    "Acc",   "SLIM 0.64 / SwiFT 0.62"),
    ("ABIDE", "Sex",       "test_f1",  "ABIDE · Sex",    "F1",    "LCM 0.87"),
    ("ADNI",  "NC_vs_MCI", "test_acc", "ADNI · NC/MCI",  "Acc",   "Brain-JEPA 0.77 / BNT 0.79"),
    ("ADNI",  "NC_vs_MCI", "test_f1",  "ADNI · NC/MCI",  "F1",    "Brain-JEPA 0.86"),
    ("ADNI",  "AD_vs_HC",  "test_auc", "ADNI · AD/HC",   "AUC",   "BrainGFM 0.80"),
    ("ADNI",  "AD_vs_HC",  "test_acc", "ADNI · AD/HC",   "Acc",   "BrainGFM 0.85"),
    ("ADNI",  "AD_vs_HC",  "test_f1",  "ADNI · AD/HC",   "F1",    "LCM 0.85"),
    ("HCP",   "Sex",       "test_acc", "HCP · Sex",      "Acc",   "SLIM 0.91"),
    ("HCP",   "Sex",       "test_f1",  "HCP · Sex",      "F1",    "SLIM 0.91 / LCM 0.73"),
]


def load(run, dataset, head="linear"):
    suf = "" if head == "linear" else f"_{head}"
    ver = RUN_VERSION.get(run, "v2")
    p = f"{LAB}/runs/{ver}/{run}/probe_{dataset.lower()}{suf}.json"
    if not os.path.exists(p):
        return {}
    return json.load(open(p)).get("results", {})


def cell(results, label, metric):
    r = (results.get(label) or {})
    v = r.get(metric)
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  -"


def sota_table():
    cache = {(run, ds): load(run, ds) for run in RUNS for ds in {a[0] for a in AXES}}
    hdr = f"  {'axis':16}{'metric':>7}" + "".join(f"{r:>9}" for r in RUNS) + "   SOTA (same metric)"
    print("\n===== v1 — SOTA-matched comparison (test = held-out 30%, linear probe) =====")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ds, label, metric, axis, mname, sota in AXES:
        cells = "".join(f"{cell(cache[(r, ds)], label, metric):>9}" for r in RUNS)
        print(f"  {axis:16}{mname:>7}{cells}   {sota}")


def head_ablation(run="base"):
    """Point-3 ablation: linear vs MLP head, on one run."""
    heads = ["linear", "mlp"]
    cache = {(h, ds): load(run, ds, h) for h in heads for ds in {a[0] for a in AXES}}
    if not any(cache[("mlp", ds)] for ds in {a[0] for a in AXES}):
        print(f"\n(point-3 ablation: no *_mlp.json yet for run '{run}' — "
              f"run  HEAD=mlp RUN={run} DATASET=... probe.sh)")
        return
    hdr = f"  {'axis':16}{'metric':>7}{'linear':>9}{'mlp':>9}   delta"
    print(f"\n===== Point-3 ablation — linear probe vs MLP head (run={run}) =====")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ds, label, metric, axis, mname, _ in AXES:
        lin = (cache[("linear", ds)].get(label) or {}).get(metric)
        mlp = (cache[("mlp", ds)].get(label) or {}).get(metric)
        d = f"{mlp - lin:+.2f}" if isinstance(lin, (int, float)) and isinstance(mlp, (int, float)) else " -"
        print(f"  {axis:16}{mname:>7}{cell(cache[('linear', ds)], label, metric):>9}"
              f"{cell(cache[('mlp', ds)], label, metric):>9}   {d}")


def main():
    sota_table()
    head_ablation("base")
    print("\n  (test metrics on the 30% held-out subjects; head hyperparam chosen by CV on train.)")


if __name__ == "__main__":
    main()
