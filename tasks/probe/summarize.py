"""Aggregate all probe_*.json across runs into comparison tables.

Usage:
    python tasks/probe/summarize.py
"""

import json
import os

LAB = "/sci/labs/arieljaffe/dan.abergel1"
RUNS = ["baseline", "fourier", "highlr", "freezeC", "freezeC_noADNI"]
FILES = [
    ("ABIDE  (TEST_auc/acc)", "probe_abide.json"),
    ("ADNI fixed (TEST_auc/acc)", "probe_adni.json"),
    ("ADNI k-fold (auc_mean±std / acc_mean±std)", "probe_adni_kfold5.json"),
]


def cell(res):
    if not res:
        return "-"
    if "test_auc" in res:                       # fixed split
        return f"{res['test_auc']:.2f}/{res.get('test_acc', 0):.2f}"
    return f"{res['auc_mean']:.2f}±{res['auc_std']:.2f}"  # k-fold


def main():
    for title, fn in FILES:
        data = {}
        for r in RUNS:
            p = f"{LAB}/runs/fmri_v2_{r}/{fn}"
            if os.path.exists(p):
                data[r] = json.load(open(p)).get("results", {})
        if not data:
            continue
        print(f"\n===== {title} =====")
        labels = list(dict.fromkeys(l for d in data.values() for l in d))
        print("  " + f"{'label':16}" + "".join(f"{r:>16}" for r in data))
        for lab in labels:
            row = "  " + f"{lab:16}"
            for r in data:
                row += f"{cell((data[r] or {}).get(lab)):>16}"
            print(row)


if __name__ == "__main__":
    main()
