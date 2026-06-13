---
title: "SOTA fMRI — synthèse pour Ariel & Yoni"
date: "2026-06-14"
geometry: margin=2cm
fontsize: 10pt
---

# Nos chiffres (Acc / F1, MAE pour les régressions)

Linear probe sur CLS gelé, 5-fold subject-aware CV. Best across iters.

| Tâche                      | A (full FT)   | C (freeze-fmri)   |
|----------------------------|:-------------:|:-----------------:|
| HCP Sex                    | 63.6 / 45.8   | **76.9 / 73.0**   |
| ADNI Sex                   | 67.0 / 59.5   | **73.9 / 71.3**   |
| ADNI Age MAE               | 5.19          | 5.70              |
| ADNI MMSE MAE              | **2.39**      | 2.75              |
| ADNI CDR_Binary            | 61.0 / 0.6    | 61.5 / 21.1       |
| ADNI CDR NC-vs-AD          | **70.8 / 0.0**| —                 |
| ADNI CDR NC-vs-MCI         | 61.1 / 75.9   | —                 |
| ADNI FAQ                   | 88.2 / 0.0    | —                 |
| ADNI Degradation 1Y/2Y/3Y  | 81.6 / 70.6 / 68.8 (F1 ~ 0) | 81.4 / 68.4 / 66.4 |

---

# Les 9 SOTA

## 1. Brain-JEPA — PUBLIÉ NeurIPS 2024 Spotlight

**arXiv** : [2409.19407](https://arxiv.org/abs/2409.19407)
**Architecture** : JEPA (Joint Embedding Predictive) + ViT sur **ROI time-series** (450 ROI × 160 timesteps). Le predictor prédit les **embeddings** de blocks cibles, pas les pixels (paradigme LeCun).
**Pretraining** : UK Biobank, 32 130 sujets (1 dataset)

| Dataset       | Tâche               | Acc       | F1        |
|---------------|---------------------|:---------:|:---------:|
| UK Biobank    | Sex                 | **88.17** | **88.58** |
| HCP-Aging     | Sex                 | **81.52** | **84.26** |
| ADNI (n=189)  | NC vs MCI           | **76.84** | **86.32** |
| ADNI          | Amyloid β+/-        | 71.00     | 75.97     |
| MACC          | NC vs MCI (Asia)    | 65.98     | 64.67     |
| OASIS-3       | AD Conversion       | 69.00     | 67.32     |
| CamCAN        | Depression          | 72.73     | 67.45     |

---

## 2. BrainLM — PUBLIÉ ICLR 2024 Spotlight

**arXiv** : [biorxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)
**Architecture** : ViT en mode **Masked Autoencoder (MAE)** sur ROI time-series. Masque 75% des patches, reconstruit les **valeurs brutes**. 111M params.
**Pretraining** : UK Biobank (76 296 rec) + HCP (1 002 rec) = 77 298 recordings / 6 700 h (2 datasets)

| Dataset       | Tâche                  | MSE z-scoré |
|---------------|------------------------|:-----------:|
| UK Biobank    | Age (régression)       | **0.503**   |
| UK Biobank    | PTSD (PCL-5)           | 0.015       |
| UK Biobank    | Anxiety (GAD-7)        | 0.073       |
| UK Biobank    | Neuroticism            | 0.069       |

**Aucun chiffre Sex / MCI / AD / MMSE rapporté.**

---

## 3. SLIM-Brain — NON PUBLIÉ (preprint déc 2025, sous review OpenReview)

**arXiv** : [2512.21881](https://arxiv.org/abs/2512.21881)
**Architecture** : **4D Hiera-JEPA** = Hiera (ViT hiérarchique pyramidal, multi-stages) + paradigme JEPA. Input = **volumes 4D `(T, X, Y, Z)`**. Tubelets 4D, attention globale par stage.
**Pretraining** : HCP + CHCP + AOMIC PIOP1 + AOMIC PIOP2 + ABCD = 4 129 sessions (5 datasets)

| Dataset      | Tâche                | Acc       | F1 / autre  |
|--------------|----------------------|:---------:|:-----------:|
| HCP          | Sex                  | **91.1**  | F1 **91.1** |
| HCP          | Fingerprint          | 98.5      | F1 98.1     |
| ADNI         | MCI classification   | **69.12** | F1 68.96    |
| ADHD-200     | TDAH                 | 63.53     | —           |
| PPMI         | Parkinson            | 70.40     | —           |
| ABIDE        | Age (classification) | 64.41     | —           |
| ABIDE        | Age (régression)     | —         | MSE 0.2175  |

*Bat Brain-JEPA sur HCP Sex (87.1) et ADNI MCI (64.53).*

---

## 4. BrainGFM — NON PUBLIÉ (preprint juin 2025)

**arXiv** : [2506.02044](https://arxiv.org/abs/2506.02044)
**Architecture** : **Graph Foundation Model** sur connectomes. Combine **Graph Contrastive Learning + Graph Masked AE** + meta-learning + graph/language prompts. Multi-atlas (8 parcellations).
**Pretraining** : 27 datasets, 25 pathologies, 25 000+ sujets / 60 000 scans

| Dataset          | Tâche               | AUC      | Acc      |
|------------------|---------------------|:--------:|:--------:|
| ADHD-200         | TDAH                | 70.6     | 72.2     |
| ABIDE II         | Autisme (ASD)       | 71.2     | 73.5     |
| **ADNI 2**       | **Alzheimer (AD)**  | **80.3** | **85.1** |
| HBN              | Dépression maj.     | 83.6     | 85.5     |
| HBN              | Anxiété             | 85.2     | 86.3     |
| HBN              | OCD                 | 80.4     | 85.8     |
| HBN              | PTSD                | 83.2     | 86.3     |
| SubMex_CUD       | Cocaïne             | 71.1     | 74.6     |
| UCLA_CNP         | Schizophrénie       | 84.2     | 86.7     |
| UCLA_CNP         | Bipolaire           | 73.5     | 76.3     |

**Aucun chiffre Sex / Age / Parkinson rapporté.**

---

## 5. LCM (Large Connectome Model) — PUBLIÉ AAAI 2026

**arXiv** : [2510.18910](https://arxiv.org/abs/2510.18910)
**Architecture** : **Transformer decoder-only** (style GPT) sur connectomes (FC matrix). MHSA + MHCA entre features connectome et tokens "brain-environment-interaction" (covariables type sexe/âge/scanner).
**Pretraining** : HCPA (713s/4 863sc) + HCPYA (248s/3 293sc) + ADNI (138) + PPMI (209) + ABIDE (1 025) + Taowu (40) + Neurocon (41) = ~10 036 scans (7 datasets)

| Dataset       | Tâche             | F1                |
|---------------|-------------------|:-----------------:|
| HCP-Aging     | Sex               | **73.94 ± 2.45**  |
| HCP-YA        | Sex               | **72.23 ± 1.92**  |
| ABIDE         | Sex               | 87.34 ± 4.48      |
| **ADNI**      | **Alzheimer**     | **85.33 ± 7.35**  |
| PPMI          | Parkinson         | 84.18 ± 11.63     |
| ABIDE         | Autisme           | 72.50 ± 1.91      |

*Subject-aware 5-fold CV (même protocole que nous).*

---

## 6. BNT (Brain Network Transformer) — PUBLIÉ NeurIPS 2022

**arXiv** : [2210.06681](https://arxiv.org/abs/2210.06681)
**Architecture** : Transformer pur sur graphes de connectome (pas un GNN). Chaque ROI = un token, features = profil de connexion. Innovation = **Orthonormal Clustering Readout** (pool par K prototypes orthogonaux appris).
**Pretraining** : AUCUN — supervisé end-to-end

| Dataset    | Tâche       | Métrique  | Score        |
|------------|-------------|-----------|:------------:|
| ABIDE      | Autisme     | AUROC     | **80.2%**    |
| ABCD       | Sex (n=7 901) | —       | non extrait  |
| ADNI       | NC vs MCI   | Acc       | **78.90**    |

*BNT bat Brain-JEPA sur ADNI NC/MCI (78.90 vs 76.84) — rapporté comme comparator dans Brain-JEPA.*

---

## 7. OViTAD — PUBLIÉ Brain Sciences 2023 (journal MDPI, pas top venue)

**arXiv** : [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
**Architecture** : **ViT-B/16 standard** sur **slices 2D** fMRI individuelles (pas de 3D, pas de temps). Inférence par **majority vote** au niveau sujet. Aucune innovation architecturale (juste tuning hyperparams).
**Pretraining** : AUCUN — supervisé end-to-end

| Dataset       | Tâche      | F1               |
|---------------|------------|:----------------:|
| ADNI (n=284)  | AD vs HC   | **0.99 ± 0.02**  |
| ADNI (n=284)  | HC vs MCI  | **0.97 ± 0.03**  |

*Test set = **31 sujets seulement** → σ=0.02 sur 31 = bruit massif. Méfiance.*

---

## 8. BrainNetCNN — PUBLIÉ NeuroImage 2017

**arXiv** : [PDF Kawahara 2017](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
**Architecture** : CNN avec **3 filtres custom topo-conscients** sur matrice de connectivité — Edge-to-Edge (E2E, agrège ligne+colonne par cellule), Edge-to-Node (E2N, agrège connexions par ROI), Node-to-Graph (N2G, pool global).
**Pretraining** : AUCUN — supervisé end-to-end

| Dataset                  | Tâche             | Score              |
|--------------------------|-------------------|:------------------:|
| DTI préterm (papier orig)| Bayley-III scores | **non comparable** |

*Pas de fMRI adulte dans le papier original. Archi réutilisée comme baseline supervisée par BNT / BrainGFM / BrainGB.*

---

## 9. SwiFT — PUBLIÉ NeurIPS 2023

**arXiv** : [2307.05916](https://arxiv.org/abs/2307.05916)
**Architecture** : **Swin Transformer 4D** sur volumes fMRI directs (sans atlas). Fenêtres 4D `(X, Y, Z, T)` + windowed self-attention + shifted windows entre layers + hiérarchique (4 stages). **L'archi la plus proche de la nôtre.**
**Pretraining** : SSL contrastif léger (composition non précisée)

| Dataset      | Tâche             | Acc       | Autre        |
|--------------|-------------------|:---------:|:------------:|
| OASIS-3      | AD Conversion     | 65.00     | F1 66.80     |
| ADNI         | MCI               | 64.45     | —            |
| ABIDE        | Age (cls)         | 62.22     | MSE 0.4137   |
| ADHD-200     | TDAH              | 60.81     | —            |
| PPMI         | Parkinson         | 58.10     | —            |
| HCP / ABCD / UKB | Sex, Age      | —         | non extraits |

*Chiffres rapportés comme comparator dans Brain-JEPA Table 9 et SLIM-Brain Table 1, pas extraits du papier SwiFT original.*

---

# Synthèse : où on se situe

1. **HCP Sex** — Nous **76.9 / 73.0 (C)**. SLIM-Brain **91.1 / 91.1** (gap 14 / 18 pts). Brain-JEPA HCPA **81.5 / 84.3** (gap 5 / 11). LCM HCPYA **F1 72** → **on est au niveau de LCM**.
2. **ADNI NC vs AD** — Nous **70.8 / 0.0** (F1 collapse). BrainGFM **AUC 80.3 / Acc 85.1** sur ADNI 2 (gap 14 pts Acc). LCM **F1 85.3**.
3. **ADNI NC vs MCI** — Nous **61.1 / 75.9**. Brain-JEPA **76.84 / 86.32** (gap 16 / 10). SLIM-Brain **69.12** Acc.
4. **ADNI Age MAE** — Nous **5.19 années**. **Aucun SOTA avec MAE en années publié** (niche).
5. **AD Conversion / Degradation** — Nous Deg1Y **81.6 / 0.0**. Brain-JEPA OASIS-3 **69.0 / 67.3**. Acc supérieure mais F1 nul.

---

# Problème critique de labels

**Nos labels CDR (`Global CDR`) ne sont PAS équivalents aux labels DX (`Diagnosis`) des SOTA.**

- BrainGFM "AD" sur ADNI 2 utilise **DX clinique** (NL / MCI / AD)
- Brain-JEPA NC/MCI utilise **DX clinique**
- Nous : **`Global CDR == 0` vs `>= 1`** → **proche mais pas identique**

Conséquence : les chiffres ci-dessus ne sont **pas directement comparables** sans aligner les labels.

---

# Top 3 SOTA à comparer en priorité

| # | Papier | Pourquoi | Notre comparator |
|:-:|---|---|---|
| 1 | **SLIM-Brain** | Même paradigme volumétrique (4D Hiera-JEPA), pretraining modeste (4k sessions) | HCP Sex 91.1 vs notre 76.9 |
| 2 | **Brain-JEPA** | Le plus comprehensive sur ADNI (NC/MCI, Amyloid, OASIS-3 AD Conv) | ADNI NC/MCI 76.84 vs 61.1 |
| 3 | **SwiFT** | Archi la plus proche (Swin 4D volumes), supervisé = baseline minimale | OASIS-3 65.0 vs nous Deg1Y 81.6 (Acc) |

---

# La direction pour la suite

**Au lieu de comparer nos labels propriétaires à leurs labels DX, télécharger les benchmarks publics qu'ils utilisent et tourner notre modèle dessus.** Comparaison archi-vs-archi sur mêmes données, mêmes labels.

| Benchmark | Accès | Utilisé par | Tâche |
|---|---|---|---|
| **ABIDE I/II** | **OPEN direct** | BNT, Brain-JEPA, BrainGFM, SLIM-Brain | Autisme |
| **ADHD-200** | OPEN direct | BrainGFM, SLIM-Brain | TDAH |
| **OASIS-3** | Open + registration | Brain-JEPA Table 9 | AD Conversion |
| **HBN** | Open + registration | BrainGFM | Dépression, OCD, PTSD |
| ADNI 2 (DX) | DUA simple | BrainGFM, Brain-JEPA | NC / MCI / AD |
| UK Biobank | DUA payant (~$1k, 2-6 mois) | Brain-JEPA, BrainLM, SwiFT | Sex, Age |

**Recommandation : commencer par ABIDE** (open download, 4 comparators d'un coup).
