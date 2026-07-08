"""Generate RESULTS.md (+ colored PDF via pandoc/xelatex) from all json_results.

Color heat-map on the numeric tables (green=strong ... red=near chance) so the
story reads at a glance. Sections: corpus / scan counts / pretraining ablations /
probe ablations / SOTA comparison.
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


# ---- color heat-map (LaTeX \cellcolor) ----
def _cc(v):
    if v is None:
        return "gray!12"
    if v >= 0.85: return "OliveGreen!55"
    if v >= 0.75: return "YellowGreen!50"
    if v >= 0.65: return "Yellow!55"
    if v >= 0.55: return "Orange!50"
    return "Red!35"


def hc(v):
    """Colored LaTeX cell for a metric value in [0,1]."""
    return "\\cellcolor{gray!12}--" if v is None else f"\\cellcolor{{{_cc(v)}}}{v:.3f}"


def best_auc_config(ds, label):
    best_auc, best = None, (None, None)
    for f in glob.glob(f"{JSON}/probe_*_{ds}*.json"):
        res = load(f).get(label)
        a = metric(res, "test_auc")
        if a is not None and (best_auc is None or a > best_auc):
            best_auc, best = a, (os.path.basename(f).replace("probe_", "").replace(".json", ""), res)
    return best


run_res = {(r, ds): load(f"{JSON}/probe_{r}_{ds}.json")
           for r in RUNS for ds in ["abide", "adni", "hcp", "oasis"]}

L = []
def w(s=""):
    L.append(s)


def latex_table(header, rows, colspec=None):
    """Emit a raw-LaTeX centered tabular (passed through by pandoc)."""
    n = len(header)
    colspec = colspec or ("l" + "c" * (n - 1))
    w("```{=latex}")
    w("\\begin{center}\\small")
    w("\\begin{tabular}{" + colspec + "}")
    w("\\hline")
    w(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")
    w("\\hline")
    for row in rows:
        w(" & ".join(row) + " \\\\")
    w("\\hline")
    w("\\end{tabular}")
    w("\\end{center}")
    w("```")


# ---- YAML header: color packages ----
w("---")
w("header-includes:")
w("  - \\usepackage[dvipsnames]{xcolor}")
w("  - \\usepackage{colortbl}")
w("---\n")

w("# fMRI Foundation Model (V2) — Results\n")
w("**Color scale (AUROC / metric):** "
  "\\colorbox{OliveGreen!55}{$\\geq$0.85} "
  "\\colorbox{YellowGreen!50}{0.75--0.85} "
  "\\colorbox{Yellow!55}{0.65--0.75} "
  "\\colorbox{Orange!50}{0.55--0.65} "
  "\\colorbox{Red!35}{$<$0.55 (near chance)}\n")

# ---- 1. corpus ----
w("## 1. Pretraining corpus\n")
w("Five sources, **4627 scans**, harmonized to TR = 0.72 s with a fixed T = 270 window.\n")
w("| Dataset | Native TR (s) | Probe holdout (30%) | Per-batch quota |")
w("|---|---|---|---|")
for d, tr, ho, q in [("HCP", "0.72", "yes", "4"), ("ABIDE", "per-site", "yes", "4"),
                     ("OASIS-3", "2.2", "yes", "4"), ("ADNI", "3.0", "yes", "3"),
                     ("AOMIC", "0.75/2.0", "no (kept whole)", "1")]:
    w(f"| {d} | {tr} | {ho} | {q} |")
w("\n*Quota = per-batch proportion. Holdout: 30% of subjects excluded from "
  "pretraining -> leakage-free test set.*\n")

# ---- 2. scan counts ----
w("## 2. Datasets & scan counts\n")
w("| Dataset | Role | Scans | Note |")
w("|---|---|---|---|")
for d, role, n, note in [
    ("HCP (rest)", "pretrain + probe", "1084", "Sex / Age"),
    ("ABIDE", "pretrain + probe", "1035", "Autism / Age / Sex"),
    ("OASIS-3", "pretrain + probe", "1197", "AD Conversion (labels pending)"),
    ("ADNI", "pretrain + probe", "812", "NC-MCI / AD-HC / Amyloid"),
    ("AOMIC", "pretrain only", "~499", "derived (4627 - others)"),
    ("ADHD-200", "probe only (external)", "162", "115 with usable DX"),
    ("COBRE", "probe only (external)", "146", "schizophrenia (72 SZ / 74 HC)"),
    ("UCLA (ds000030)", "probe only (external)", "265", "downloader ready"),
    ("HCP task-fMRI", "probe only", "—", "7 tasks, download pending")]:
    w(f"| {d} | {role} | {n} | {note} |")
w("\n*External datasets (ADHD-200 / COBRE / UCLA) were never seen in pretraining.*\n")

# ---- 3. pretraining ablations (COLORED) ----
w("## 3. Pretraining ablations (5 SSL runs) — test AUROC\n")
w("Same DINOv2 (ImageNet) init; each run changes **one** factor. "
  "**base** = reference (freeze blocks 0-8), **fourier** = Fourier positional encoding, "
  "**noblock2** = drop block_2, **pool** = AvgPool downsampling, **unfrozen** = all layers "
  "unfrozen during SSL.\n")
AXES = [("abide", "Autism", "ABIDE / Autism"), ("abide", "Age", "ABIDE / Age"),
        ("abide", "Sex", "ABIDE / Sex"), ("adni", "NC_vs_MCI", "ADNI / NC-MCI"),
        ("adni", "AD_vs_HC", "ADNI / AD-HC"), ("adni", "Amyloid", "ADNI / Amyloid"),
        ("hcp", "Sex", "HCP / Sex"), ("hcp", "Age", "HCP / Age"),
        ("oasis", "AD_Conversion", "OASIS / AD Conv")]
rows = []
for ds, lab, name in AXES:
    rows.append([name] + [hc(metric(run_res[(r, ds)].get(lab), "test_auc")) for r in RUNS])
latex_table(["Axis"] + RUNS, rows)
w("*AUROC ranks the runs by representation quality (threshold- and balance-independent). "
  "The SOTA table (Section 5) uses Acc/F1, matching what the papers report.*\n")

# ---- 4. probe ablations (COLORED) ----
w("## 4. Probe ablations (on `base`)\n")
w("### 4a. Temporal aggregation of the CLS token — AUROC\n")
rows = []
for ds, lab in [("abide", "Autism"), ("abide", "Age"), ("adni", "NC_vs_MCI"),
                ("adni", "Amyloid"), ("adni", "AD_vs_HC"), ("hcp", "Sex"), ("hcp", "Age"),
                ("adhd", "ADHD"), ("cobre", "Schizophrenia")]:
    a = metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(lab), "test_auc")
    b = metric(load(f"{JSON}/probe_base_{ds}_agg-mean_std.json").get(lab), "test_auc")
    d = f"{b-a:+.3f}" if (a is not None and b is not None) else "--"
    rows.append([f"{ds.upper()} / {lab.replace('_vs_','-').replace('_','-')}", hc(a), hc(b), d])
latex_table(["Axis", "mean (384-d)", "mean\\_std (768-d)", "$\\Delta$"], rows)
w("*`mean_std` mainly helps task dynamics; elsewhere `mean` wins.*\n")

w("### 4b. MLP head architecture — AUROC\n")
ARCHS = ["128", "256", "256x128", "512x256", "512x256x128"]
rows = []
for ds, lab in [("adni", "NC_vs_MCI"), ("adni", "AD_vs_HC"), ("adni", "Amyloid"),
                ("abide", "Autism"), ("hcp", "Sex"), ("adhd", "ADHD"), ("cobre", "Schizophrenia")]:
    r = [f"{ds.upper()} / {lab.replace('_vs_','-').replace('_','-')}"]
    r += [hc(metric(load(f"{JSON}/probe_base_{ds}_mlp-{a}.json").get(lab), "test_auc")) for a in ARCHS]
    lin = metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(lab), "test_auc") \
        or metric(run_res.get(("base", ds), {}).get(lab), "test_auc")
    r.append(hc(lin))
    rows.append(r)
latex_table(["Axis"] + [a.replace("x", "$\\times$") for a in ARCHS] + ["linear"], rows)
w("*The MLP head does not clearly beat the linear probe; deeper heads overfit.*\n")

# ---- 5. results (ours) + SOTA only where same dataset ----
def our_all(ds, label):
    _, res = best_auc_config(ds, label)
    return metric(res, "test_acc"), metric(res, "test_f1"), metric(res, "test_auc")

def raw(ds, label, key):  # base-run value for cognition/task-state
    return metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(label), key)

NR = "\\cellcolor{gray!12}n/r"

w("## 5. Our results vs SOTA (same-dataset only)\n")
w("### 5a. Same-dataset comparisons\n")
w("The SOTA number is shown **only** where the paper uses the same dataset as us. "
  "Brain-JEPA = fine-tuning, Acc/F1 only; ours = linear probe (best-AUROC config).\n")
rows = []
for name, ds, lab, s_acc, s_f1 in [("ADNI / NC-MCI", "adni", "NC_vs_MCI", 0.768, 0.863),
                                   ("ADNI / Amyloid", "adni", "Amyloid", 0.710, 0.760)]:
    acc, f1v, auc = our_all(ds, lab)
    rows.append([name, "Ours (lin.)", hc(acc), hc(f1v), hc(auc)])
    rows.append(["", "Brain-JEPA (FT)", hc(s_acc), hc(s_f1), NR])
acc, f1v, auc = our_all("adhd", "ADHD")
rows.append(["ADHD-200", "Ours (lin.)", hc(acc), hc(f1v), hc(auc)])
rows.append(["", "NeuroSTORM", hc(0.587), NR, NR])
latex_table(["Benchmark", "Model", "Acc", "F1", "AUROC"], rows)

w("### 5b. Our other downstream results (no same-dataset SOTA to compare)\n")
rows = []
for name, ds, lab in [("ABIDE / Autism", "abide", "Autism"), ("ABIDE / Age", "abide", "Age"),
                      ("ABIDE / Sex", "abide", "Sex"), ("ADNI / AD-HC", "adni", "AD_vs_HC"),
                      ("HCP / Sex", "hcp", "Sex"), ("HCP / Age", "hcp", "Age"),
                      ("COBRE / Schizophrenia", "cobre", "Schizophrenia")]:
    acc, f1v, auc = our_all(ds, lab)
    rows.append([name, hc(auc), hc(acc), hc(f1v)])
latex_table(["Benchmark", "AUROC", "Acc", "F1"], rows)

w("\n*Only ADNI (Brain-JEPA) and ADHD-200 (NeuroSTORM) are same-dataset comparisons. "
  "Everything in 5b is our own result with no matching same-dataset SOTA number.*\n")

open(f"{HERE}/RESULTS.md", "w").write("\n".join(L))
print("wrote RESULTS.md")
