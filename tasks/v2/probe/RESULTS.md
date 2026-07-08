# fMRI Foundation Model (V2) — Résultats

## 1. Corpus de pré-entraînement

5 sources, **4627 scans**, harmonisées à TR=0.72 s, fenêtre T=270.

| Dataset | TR natif (s) | Holdout probe (30 %) | Quota/batch |
|---|---|---|---|
| HCP | 0.72 | oui | 4 |
| ABIDE | par-site | oui | 4 |
| OASIS-3 | 2.2 | oui | 4 |
| ADNI | 3.0 | oui | 3 |
| AOMIC | 0.75/2.0 | non (entier) | 1 |

*Le quota = proportion par batch (échantillonnage proportionnel). Holdout : 30 % des sujets exclus du pré-entraînement → test sans leakage.*

## 2. Ablations de pré-entraînement (5 runs SSL) — AUC test

Chaque run part du même init DINOv2 (ImageNet), diffère par **un** facteur :

- **base** : référence (freeze blocs 0–8) · **fourier** : encodage positionnel de Fourier
- **noblock2** : suppression de block_2 · **pool** : downsampling par AvgPool (au lieu de stride)
- **unfrozen** : toutes les couches dégelées pendant le SSL

| Axe | base | fourier | noblock2 | pool | unfrozen |
|---|---|---|---|---|---|
| ABIDE · Autism | 0.605 | 0.518 | 0.484 | 0.606 | 0.541 |
| ABIDE · Age | 0.846 | 0.756 | 0.776 | 0.877 | 0.828 |
| ABIDE · Sex | 0.521 | 0.514 | 0.492 | 0.694 | 0.538 |
| ADNI · NC/MCI | 0.567 | 0.497 | 0.484 | 0.566 | 0.604 |
| ADNI · AD/HC | 0.481 | 0.537 | 0.577 | 0.591 | 0.731 |
| ADNI · Amyloid | 0.611 | 0.539 | 0.500 | 0.641 | 0.640 |
| HCP · Sex | 0.908 | 0.900 | 0.933 | 0.962 | 0.886 |
| HCP · Age | 0.633 | 0.594 | 0.626 | 0.680 | 0.592 |
| OASIS · AD Conv | — | — | — | — | — |

*Métrique : AUROC test (probe linéaire, 30 % held-out). OASIS = labels absents.*

## 3. Ablations de probe (sur `base`)

### 3a. Agrégation temporelle du CLS — AUC

| Axe | mean (384-d) | mean_std (768-d) | Δ |
|---|---|---|---|
| ABIDE · Autism | 0.605 | 0.583 | -0.023 |
| ABIDE · Age | 0.846 | 0.835 | -0.011 |
| ADNI · NC_vs_MCI | 0.567 | 0.505 | -0.062 |
| ADNI · Amyloid | 0.611 | 0.606 | -0.005 |
| ADNI · AD_vs_HC | 0.481 | 0.404 | -0.077 |
| HCP · Sex | 0.908 | 0.853 | -0.055 |
| HCP · Age | 0.633 | 0.595 | -0.037 |
| ADHD · ADHD | 0.552 | 0.557 | +0.005 |
| COBRE · Schizophrenia | 0.519 | 0.538 | +0.019 |

*mean_std aide surtout le task-state (dynamique) ; ailleurs `mean` domine.*

### 3b. Architecture de tête MLP — AUC

| Axe | 128 | 256 | 256x128 | 512x256 | 512x256x128 | linéaire |
|---|---|---|---|---|---|---|
| ADNI · NC_vs_MCI | 0.588 | 0.509 | 0.523 | 0.475 | 0.433 | 0.567 |
| ADNI · AD_vs_HC | 0.407 | 0.467 | 0.472 | 0.448 | 0.487 | 0.481 |
| ADNI · Amyloid | 0.621 | 0.623 | 0.610 | 0.631 | 0.610 | 0.611 |
| ABIDE · Autism | 0.550 | 0.497 | 0.561 | 0.509 | 0.522 | 0.605 |
| HCP · Sex | 0.844 | 0.906 | 0.896 | 0.898 | 0.895 | 0.908 |
| ADHD · ADHD | 0.477 | 0.466 | 0.382 | 0.412 | 0.397 | 0.552 |
| COBRE · Schizophrenia | 0.602 | 0.566 | 0.595 | 0.557 | 0.590 | 0.519 |

*Le MLP n'améliore pas nettement le linéaire ; les archis profondes surapprennent.*

## 4. Meilleurs résultats vs SOTA (comparaisons à dataset identique)

Réserves : (1) les chiffres Brain-JEPA sont en **fine-tune**, les nôtres en **linear probe** ; (2) on reporte l'acc/F1 de la config au **meilleur AUROC** (pas le max d'acc brut, qui récompenserait un classifieur de classe majoritaire).

### Brain-JEPA (même dataset = ADNI)

| Benchmark | Métrique | Nous (best-AUC) | config | Brain-JEPA (FT) |
|---|---|---|---|---|
| ADNI · NC/MCI | Acc | 0.576 | unfrozen_adni | 0.768 |
| ADNI · NC/MCI | F1 | 0.632 | unfrozen_adni | 0.863 |
| ADNI · Amyloid | Acc | 0.559 | pool_adni | 0.710 |
| ADNI · Amyloid | F1 | 0.623 | pool_adni | 0.760 |

### NeuroSTORM (même dataset = ADHD-200)

| Benchmark | Métrique | Nous (best-AUC) | config | NeuroSTORM |
|---|---|---|---|---|
| ADHD-200 | Acc | 0.617 | base_adhd_agg-mean_std | 0.587 |
| ADHD-200 | AUROC | 0.557 | base_adhd_agg-mean_std | — |

**Non comparés (dataset différent)** : HCP Sex/Age (HCP-YA vs HCP-Aging), COBRE vs HCP-EP, UCLA (à télécharger). OASIS/ABIDE ne sont pas des benchmarks Brain-JEPA.


**Sources SOTA** : Brain-JEPA (arXiv 2409.19407, Tables 2-3, fine-tune) · NeuroSTORM (arXiv 2506.11167).
