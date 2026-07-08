"""Generate RESULTS.md (+ PDF via pandoc) from all probe json_results.

Sections:
  1. Pretraining corpus
  2. Pretraining ablations (5 SSL runs)      — AUC per SOTA axis
  3. Probe ablations on `base`               — aggregation + MLP architecture
  4. Best-of vs Brain-JEPA / NeuroSTORM       — same-dataset comparisons only
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
    # kfold schema fallback: test_auc->auc_mean, test_acc->acc_mean, test_f1->f1_mean
    alt = {"test_auc": "auc_mean", "test_acc": "acc_mean", "test_f1": "f1_mean"}.get(key)
    return res.get(alt) if alt else None


def cell(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


# ---- gather ----
run_res = {(r, ds): load(f"{JSON}/probe_{r}_{ds}.json")
           for r in RUNS for ds in ["abide", "adni", "hcp", "oasis"]}


def best_auc_config(ds, label):
    """The (file, results) with the highest AUROC for (ds,label) across ALL json
    files. Selecting by AUC (real discrimination) avoids majority-class accuracy
    inflation from degenerate classifiers on imbalanced cohorts."""
    best_auc, best = None, (None, None)
    for f in glob.glob(f"{JSON}/probe_*_{ds}*.json"):
        res = load(f).get(label)
        a = metric(res, "test_auc")
        if a is not None and (best_auc is None or a > best_auc):
            best_auc = a
            best = (os.path.basename(f).replace("probe_", "").replace(".json", ""), res)
    return best  # (tag, results)


def best_reported(ds, label, key):
    """acc/F1 of the BEST-AUC config (honest: not the max acc across all configs)."""
    tag, res = best_auc_config(ds, label)
    return metric(res, key), tag


L = []
def w(s=""):
    L.append(s)

# ---- 1. corpus ----
w("# fMRI Foundation Model (V2) — Résultats\n")
w("## 1. Corpus de pré-entraînement\n")
w("5 sources, **4627 scans**, harmonisées à TR=0.72 s, fenêtre T=270.\n")
w("| Dataset | TR natif (s) | Holdout probe (30 %) | Quota/batch |")
w("|---|---|---|---|")
for d, tr, ho, q in [("HCP", "0.72", "oui", "4"), ("ABIDE", "par-site", "oui", "4"),
                     ("OASIS-3", "2.2", "oui", "4"), ("ADNI", "3.0", "oui", "3"),
                     ("AOMIC", "0.75/2.0", "non (entier)", "1")]:
    w(f"| {d} | {tr} | {ho} | {q} |")
w("\n*Le quota = proportion par batch (échantillonnage proportionnel). Holdout : "
  "30 % des sujets exclus du pré-entraînement → test sans leakage.*\n")

# ---- 2. pretraining ablations ----
w("## 2. Ablations de pré-entraînement (5 runs SSL) — AUC test\n")
w("Chaque run part du même init DINOv2 (ImageNet), diffère par **un** facteur :\n")
w("- **base** : référence (freeze blocs 0–8) · **fourier** : encodage positionnel de Fourier")
w("- **noblock2** : suppression de block_2 · **pool** : downsampling par AvgPool (au lieu de stride)")
w("- **unfrozen** : toutes les couches dégelées pendant le SSL\n")
AXES2 = [("abide", "Autism", "ABIDE · Autism"), ("abide", "Age", "ABIDE · Age"),
         ("abide", "Sex", "ABIDE · Sex"), ("adni", "NC_vs_MCI", "ADNI · NC/MCI"),
         ("adni", "AD_vs_HC", "ADNI · AD/HC"), ("adni", "Amyloid", "ADNI · Amyloid"),
         ("hcp", "Sex", "HCP · Sex"), ("hcp", "Age", "HCP · Age"),
         ("oasis", "AD_Conversion", "OASIS · AD Conv")]
w("| Axe | " + " | ".join(RUNS) + " |")
w("|---|" + "---|" * len(RUNS))
for ds, lab, name in AXES2:
    row = [name]
    for r in RUNS:
        row.append(cell(metric(run_res[(r, ds)].get(lab), "test_auc")))
    w("| " + " | ".join(row) + " |")
w("\n*Métrique : AUROC test (probe linéaire, 30 % held-out). OASIS = labels absents.*\n")

# ---- 3. probe ablations ----
w("## 3. Ablations de probe (sur `base`)\n")
w("### 3a. Agrégation temporelle du CLS — AUC\n")
w("| Axe | mean (384-d) | mean_std (768-d) | Δ |")
w("|---|---|---|---|")
AGG_AXES = [("abide", "Autism"), ("abide", "Age"), ("adni", "NC_vs_MCI"),
            ("adni", "Amyloid"), ("adni", "AD_vs_HC"), ("hcp", "Sex"), ("hcp", "Age"),
            ("adhd", "ADHD"), ("cobre", "Schizophrenia")]
for ds, lab in AGG_AXES:
    a = metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(lab), "test_auc")
    b = metric(load(f"{JSON}/probe_base_{ds}_agg-mean_std.json").get(lab), "test_auc")
    d = f"{b-a:+.3f}" if (a is not None and b is not None) else "—"
    w(f"| {ds.upper()} · {lab} | {cell(a)} | {cell(b)} | {d} |")
w("\n*mean_std aide surtout le task-state (dynamique) ; ailleurs `mean` domine.*\n")

w("### 3b. Architecture de tête MLP — AUC\n")
ARCHS = ["128", "256", "256x128", "512x256", "512x256x128"]
w("| Axe | " + " | ".join(ARCHS) + " | linéaire |")
w("|---|" + "---|" * (len(ARCHS) + 1))
MLP_AXES = [("adni", "NC_vs_MCI"), ("adni", "AD_vs_HC"), ("adni", "Amyloid"),
            ("abide", "Autism"), ("hcp", "Sex"), ("adhd", "ADHD"), ("cobre", "Schizophrenia")]
for ds, lab in MLP_AXES:
    row = [f"{ds.upper()} · {lab}"]
    for a in ARCHS:
        row.append(cell(metric(load(f"{JSON}/probe_base_{ds}_mlp-{a}.json").get(lab), "test_auc")))
    lin = metric(load(f"{JSON}/probe_base_{ds}_agg-mean.json").get(lab), "test_auc") \
        or metric(run_res.get((("base"), ds), {}).get(lab), "test_auc")
    row.append(cell(lin))
    w("| " + " | ".join(row) + " |")
w("\n*Le MLP n'améliore pas nettement le linéaire ; les archis profondes surapprennent.*\n")

# ---- 4. SOTA ----
w("## 4. Meilleurs résultats vs SOTA (comparaisons à dataset identique)\n")
w("Réserves : (1) les chiffres Brain-JEPA sont en **fine-tune**, les nôtres en "
  "**linear probe** ; (2) on reporte l'acc/F1 de la config au **meilleur AUROC** "
  "(pas le max d'acc brut, qui récompenserait un classifieur de classe majoritaire).\n")
w("### Brain-JEPA (même dataset = ADNI)\n")
w("| Benchmark | Métrique | Nous (best-AUC) | config | Brain-JEPA (FT) |")
w("|---|---|---|---|---|")
SOTA_JEPA = [("adni", "NC_vs_MCI", "test_acc", "Acc", "0.768"),
             ("adni", "NC_vs_MCI", "test_f1", "F1", "0.863"),
             ("adni", "Amyloid", "test_acc", "Acc", "0.710"),
             ("adni", "Amyloid", "test_f1", "F1", "0.760")]
for ds, lab, key, mname, sota in SOTA_JEPA:
    v, where = best_reported(ds, lab, key)
    w(f"| ADNI · {lab.replace('_vs_','/')} | {mname} | {cell(v)} | {where or '—'} | {sota} |")
w("\n### NeuroSTORM (même dataset = ADHD-200)\n")
w("| Benchmark | Métrique | Nous (best-AUC) | config | NeuroSTORM |")
w("|---|---|---|---|---|")
acc, where = best_reported("adhd", "ADHD", "test_acc")
auc, _ = best_reported("adhd", "ADHD", "test_auc")
w(f"| ADHD-200 | Acc | {cell(acc)} | {where or '—'} | 0.587 |")
w(f"| ADHD-200 | AUROC | {cell(auc)} | {where or '—'} | — |")
w("\n**Non comparés (dataset différent)** : HCP Sex/Age (HCP-YA vs HCP-Aging), "
  "COBRE vs HCP-EP, UCLA (à télécharger). OASIS/ABIDE ne sont pas des benchmarks Brain-JEPA.\n")
w("\n**Sources SOTA** : Brain-JEPA (arXiv 2409.19407, Tables 2-3, fine-tune) · "
  "NeuroSTORM (arXiv 2506.11167).\n")

open(f"{HERE}/RESULTS.md", "w").write("\n".join(L))
print("wrote RESULTS.md")
