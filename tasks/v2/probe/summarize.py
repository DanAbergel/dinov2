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
    ("ADNI",  "Amyloid",   "test_acc", "ADNI · Amyloid", "Acc",   "Brain-JEPA 0.71"),
    ("ADNI",  "Amyloid",   "test_f1",  "ADNI · Amyloid", "F1",    "Brain-JEPA 0.76"),
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


def load_ft(run, dataset, depth):
    ver = RUN_VERSION.get(run, "v2")
    p = f"{LAB}/runs/{ver}/{run}/finetune_{dataset.lower()}_{depth}.json"
    if not os.path.exists(p):
        return {}
    return json.load(open(p)).get("results", {})


def ft_cell(res, label, metric):
    ms = (res.get(label) or {}).get("mean_std") or {}
    v = ms.get(metric)
    return f"{v[0]:.2f}" if isinstance(v, (list, tuple)) and v else "  -"


def ladder(run="base"):
    """Adaptation ladder (Brain-JEPA style): frozen linear -> frozen MLP ->
    fine-tune last3 -> fine-tune all, on one run. Point 3 = the MLP step,
    point 4 = the FT steps. FT cells show the mean over seeds."""
    lin = {ds: load(run, ds, "linear") for ds in {a[0] for a in AXES}}
    mlp = {ds: load(run, ds, "mlp") for ds in {a[0] for a in AXES}}
    l3 = {ds: load_ft(run, ds, "last3") for ds in {a[0] for a in AXES}}
    al = {ds: load_ft(run, ds, "all") for ds in {a[0] for a in AXES}}
    hdr = (f"  {'axis':16}{'metric':>7}{'linear':>9}{'MLP':>9}{'FT-l3':>9}{'FT-all':>9}"
           f"   SOTA (same metric)")
    print(f"\n===== Adaptation ladder (run={run}) — frozen probe -> fine-tune =====")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ds, label, metric, axis, mname, sota in AXES:
        row = (f"  {axis:16}{mname:>7}"
               f"{cell(lin[ds], label, metric):>9}{cell(mlp[ds], label, metric):>9}"
               f"{ft_cell(l3[ds], label, metric):>9}{ft_cell(al[ds], label, metric):>9}"
               f"   {sota}")
        print(row)


def main():
    sota_table()
    ladder("base")
    print("\n  (frozen: test on 30% held-out, hyperparam by CV on train. "
          "FT: mean over seeds, 15% val' carved from train for early stopping.)")


if __name__ == "__main__":
    main()
