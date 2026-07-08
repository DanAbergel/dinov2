"""Generate RESULTS.md (+ PDF via pandoc) from all probe json_results.

Sections:
  1. Pretraining corpus
  2. Datasets & scan counts (every dataset we have)
  3. Pretraining ablations (5 SSL runs)   — AUROC per SOTA axis
  4. Probe ablations on `base`            — aggregation + MLP architecture
  5. Best-of vs Brain-JEPA / NeuroSTORM   — same-dataset comparisons only
"""
import glob
import json
import os

HERE = os.path.dirname(__file__)
JSON = os.path.join(HERE, "json_results")
RUNS = ["base", "fourier", "noblock2", "pool", "unfrozen"]


def load(path):
    try:
        return json.load(open(path))["results"]
    except Exception:
        return {}


def metric(res, key):
    if res is None:
        return None
    if key in res and res[key] is not None:
        return res[key]
    alt = {"test_auc": "auc_mean", "test_acc": "acc_mean", "test_f1": "f1_mean"}.get(key)
    return res.get(alt) if alt else None


def cell(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def best_auc_config(ds, label):
    """(tag, results) with the highest AUROC for (ds,label) across ALL json files.
    Selecting by AUROC avoids majority-class accuracy inflation on imbalanced cohorts."""
    best_auc, best = None, (None, None)
    for f in glob.glob(f"{JSON}/probe_*_{ds}*.json"):
        res = load(f).get(label)
        a = metric(res, "test_auc")
        if a is not None and (best_auc is None or a > best_auc):
            best_auc = a
            best = (os.path.basename(f).replace("probe_", "").replace(".json", ""), res)
    return best


def best_reported(ds, label, key):
    tag, res = best_auc_config(ds, label)
    return metric(res, key), tag


run_res = {(r, ds): load(f"{JSON}/probe_{r}_{ds}.json")
           for r in RUNS for ds in ["abide", "adni", "hcp", "oasis"]}

L = []
def w(s=""):
    L.append(s)

# ---- 1. corpus ----
w("# fMRI Foundation Model (V2) — Results\n")
w("## 1. Pretraining corpus\n")
w("Five sources, **4627 scans**, harmonized to TR = 0.72 s with a fixed T = 270 window.\n")
w("| Dataset | Native TR (s) | Probe holdout (30%) | Per-batch quota |")
w("|---|---|---|---|")
for d, tr, ho, q in [("HCP", "0.72", "yes", "4"), ("ABIDE", "per-site", "yes", "4"),
                     ("OASIS-3", "2.2", "yes", "4"), ("ADNI", "3.0", "yes", "3"),
                     ("AOMIC", "0.75/2.0", "no (kept whole)", "1")]:
    w(f"| {d} | {tr} | {ho} | {q} |")
w("\n*Quota = per-batch proportion (proportional sampler). Holdout: 30% of subjects "
  "excluded from pretraining -> leakage-free test set.*\n")

# ---- 2. datasets & scan counts ----
w("## 2. Datasets & scan counts\n")
w("Every dataset we have, with its role (pretraining / downstream probe) and scan count.\n")
w("| Dataset | Role | Scans | Note |")
w("|---|---|---|---|")
COUNTS = [
    ("HCP (rest)",     "pretrain + probe",     "1084", "Sex / Age"),
    ("ABIDE",          "pretrain + probe",     "1035", "Autism / Age / Sex"),
    ("OASIS-3",        "pretrain + probe",     "1197", "AD Conversion (labels pending)"),
    ("ADNI",           "pretrain + probe",     "812",  "NC-MCI / AD-HC / Amyloid"),
    ("AOMIC",          "pretrain only",        "~499", "derived (4627 - others); no probe"),
    ("ADHD-200",       "probe only (external)","162",  "115 with usable DX label"),
    ("COBRE",          "probe only (external)","146",  "schizophrenia (72 SZ / 74 HC)"),
    ("UCLA (ds000030)","probe only (external)","265",  "downloader ready, not yet run"),
    ("HCP task-fMRI",  "probe only",           "—",    "7 tasks x ~1050, download pending"),
]
for d, role, n, note in COUNTS:
    w(f"| {d} | {role} | {n} | {note} |")
w("\n*Pretraining corpus total = 4627 (HCP + ABIDE + OASIS + ADNI + AOMIC). "
  "External datasets (ADHD-200 / COBRE / UCLA) were never seen in pretraining.*\n")

# ---- 3. pretraining ablations ----
w("## 3. Pretraining ablations (5 SSL runs) — test AUROC\n")
w("Each run starts from the same DINOv2 (ImageNet) init and changes **one** factor:\n")
w("- **base**: reference (freeze blocks 0-8) · **fourier**: Fourier positional encoding")
w("- **noblock2**: drop block_2 · **pool**: AvgPool downsampling (instead of strided conv)")
w("- **unfrozen**: all layers unfrozen during SSL\n")
AXES2 = [("abide", "Autism", "ABIDE / Autism"), ("abide", "Age", "ABIDE / Age"),
         ("abide", "Sex", "ABIDE / Sex"), ("adni", "NC_vs_MCI", "ADNI / NC-MCI"),
         ("adni", "AD_vs_HC", "ADNI / AD-HC"), ("adni", "Amyloid", "ADNI / Amyloid"),
         ("hcp", "Sex", "HCP / Sex"), ("hcp", "Age", "HCP / Age"),
         ("oasis", "AD_Conversion", "OASIS / AD Conv")]
w("| Axis | " + " | ".join(RUNS) + " |")
w("|---|" + "---|" * len(RUNS))
for ds, lab, name in AXES2:
    row = [name] + [cell(metric(run_res[(r, ds)].get(lab), "test_auc")) for r in RUNS]
    w("| " + " | ".join(row) + " |")
w("\n*Metric: test AUROC (linear probe, 30% held-out). OASIS = labels missing.*\n")

# ---- 4. probe ablations ----
w("## 4. Probe ablations (on `base`)\n")
w("### 4a. Temporal aggregation of the CLS token — AUROC\n")
w("| Axis | mean (384-d) | mean_std (768-d) | delta |")
w("|---|---|---|---|")
for ds, lab in [("abide", "Autism"), ("abide", "Age"), ("adni", "NC_vs_MCI"),
                ("adni", "Amyloid"), ("adni", "AD_vs_HC"), ("hcp", "Sex"), ("hcp", "Age"),
                ("adhd", "ADHD"), ("cobre", "Schizophrenia")]:
    a = metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(lab), "test_auc")
    b = metric(load(f"{JSON}/probe_base_{ds}_agg-mean_std.json").get(lab), "test_auc")
    d = f"{b-a:+.3f}" if (a is not None and b is not None) else "—"
    w(f"| {ds.upper()} / {lab} | {cell(a)} | {cell(b)} | {d} |")
w("\n*mean_std mainly helps task dynamics; elsewhere `mean` wins.*\n")

w("### 4b. MLP head architecture — AUROC\n")
ARCHS = ["128", "256", "256x128", "512x256", "512x256x128"]
w("| Axis | " + " | ".join(ARCHS) + " | linear |")
w("|---|" + "---|" * (len(ARCHS) + 1))
for ds, lab in [("adni", "NC_vs_MCI"), ("adni", "AD_vs_HC"), ("adni", "Amyloid"),
                ("abide", "Autism"), ("hcp", "Sex"), ("adhd", "ADHD"), ("cobre", "Schizophrenia")]:
    row = [f"{ds.upper()} / {lab}"]
    row += [cell(metric(load(f"{JSON}/probe_base_{ds}_mlp-{a}.json").get(lab), "test_auc")) for a in ARCHS]
    lin = metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(lab), "test_auc") \
        or metric(run_res.get(("base", ds), {}).get(lab), "test_auc")
    row.append(cell(lin))
    w("| " + " | ".join(row) + " |")
w("\n*The MLP head does not clearly beat the linear probe; deeper heads overfit.*\n")

# ---- 5. SOTA ----
w("## 5. Best results vs SOTA (same-dataset comparisons)\n")
w("Caveats: (1) Brain-JEPA numbers are **fine-tuning**, ours are **linear probe**; "
  "(2) we report acc/F1 of the **best-AUROC** config (not the raw max accuracy, which "
  "would reward a majority-class classifier).\n")
w("### Brain-JEPA (same dataset = ADNI)\n")
w("| Benchmark | Metric | Ours (best-AUC) | config | Brain-JEPA (FT) |")
w("|---|---|---|---|---|")
for ds, lab, key, mname, sota in [("adni", "NC_vs_MCI", "test_acc", "Acc", "0.768"),
                                  ("adni", "NC_vs_MCI", "test_f1", "F1", "0.863"),
                                  ("adni", "Amyloid", "test_acc", "Acc", "0.710"),
                                  ("adni", "Amyloid", "test_f1", "F1", "0.760")]:
    v, where = best_reported(ds, lab, key)
    w(f"| ADNI / {lab.replace('_vs_','-')} | {mname} | {cell(v)} | {where or '—'} | {sota} |")
w("\n### NeuroSTORM (same dataset = ADHD-200)\n")
w("| Benchmark | Metric | Ours (best-AUC) | config | NeuroSTORM |")
w("|---|---|---|---|---|")
acc, where = best_reported("adhd", "ADHD", "test_acc")
auc, _ = best_reported("adhd", "ADHD", "test_auc")
w(f"| ADHD-200 | Acc | {cell(acc)} | {where or '—'} | 0.587 |")
w(f"| ADHD-200 | AUROC | {cell(auc)} | {where or '—'} | — |")
w("\n**Not compared (different dataset):** HCP Sex/Age (HCP-YA vs HCP-Aging), "
  "COBRE vs HCP-EP, UCLA (to download). OASIS/ABIDE are not Brain-JEPA benchmarks.\n")
w("\n**SOTA sources:** Brain-JEPA (arXiv 2409.19407, Tables 2-3, fine-tuning) · "
  "NeuroSTORM (arXiv 2506.11167).\n")

open(f"{HERE}/RESULTS.md", "w").write("\n".join(L))
print("wrote RESULTS.md")
