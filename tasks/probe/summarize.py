"""Aggregate probe_*.json across runs into SOTA-comparison tables.

For each (dataset, task) it picks the config with the best VAL AUC (the correct
leakage-free selection) and prints its TEST AUC/Acc next to the SOTA reference.
k-fold results (full-cohort) are shown as mean±std.

Usage:
    python tasks/probe/summarize.py
"""

import json
import os

LAB = "/sci/labs/arieljaffe/dan.abergel1"
RUNS = ["baseline", "fourier", "highlr", "freezeC", "freezeC_noADNI"]
FILES = [("ABIDE", "probe_abide.json", False),
         ("ADNI (fixed — small test!)", "probe_adni.json", False),
         ("ADNI (k-fold, full cohort)", "probe_adni_kfold5.json", True)]

# SOTA reference per task (from SOTA_COMPARISON_EN.pdf)
SOTA = {
    "Autism":   "BrainGFM 0.71 AUC / LCM 0.73 F1 / BNT 0.80",
    "Age":      "SLIM-Brain 0.64 / SwiFT 0.62 (acc)",
    "Sex":      "LCM 0.87 F1",
    "NC_vs_MCI": "Brain-JEPA 0.77 / BNT 0.79 (acc)",
    "AD_vs_HC": "BrainGFM 0.85 / LCM 0.85 (AUC/F1)",
}


def main():
    for title, fn, kfold in FILES:
        data = {}
        for r in RUNS:
            p = f"{LAB}/runs/fmri_v2_{r}/{fn}"
            if os.path.exists(p):
                data[r] = json.load(open(p)).get("results", {})
        if not data:
            continue
        print(f"\n===== {title} =====")
        labels = list(dict.fromkeys(l for d in data.values() for l in d))

        if kfold:
            print(f"  {'task':14}{'AUC':>14}{'Acc':>14}   SOTA")
            for lab in labels:
                # only one run (the no-ADNI) typically; show it
                for r, res in data.items():
                    v = res.get(lab)
                    if v and "auc_mean" in v:
                        print(f"  {lab:14}{v['auc_mean']:.2f}±{v['auc_std']:.2f}   "
                              f"{v['acc_mean']:.2f}±{v['acc_std']:.2f}   {SOTA.get(lab,'')}")
            continue

        print(f"  {'task':14}{'best(val)':>14}{'val':>6}{'TEST_auc':>9}{'acc':>6}   SOTA")
        for lab in labels:
            # pick config with best VAL auc
            best, bestv = None, -1
            for r, res in data.items():
                v = res.get(lab)
                if v and v.get("val_auc", -1) > bestv:
                    bestv, best = v["val_auc"], (r, v)
            if not best:
                continue
            r, v = best
            print(f"  {lab:14}{r:>14}{v['val_auc']:>6.2f}{v['test_auc']:>9.2f}"
                  f"{v.get('test_acc',0):>6.2f}   {SOTA.get(lab,'')}")


if __name__ == "__main__":
    main()
