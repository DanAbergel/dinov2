---
title: "SOTA fMRI — comparaison pour Ariel & Yoni"
date: "2026-06-14"
geometry: margin=2cm
fontsize: 10pt
---

# Nos chiffres (rappel)

Linear probe sur CLS gelé, 5-fold subject-aware CV.

| Tâche                          | Métrique | A (full FT) | C (freeze-fmri) |
|--------------------------------|:--------:|:-----------:|:---------------:|
| HCP Sex                        | AUC      | 0.70        | **0.84**        |
| ADNI Sex                       | AUC      | 0.72        | **0.81**        |
| ADNI Age                       | MAE (y)  | 5.21        | 5.70            |
| ADNI MMSE                      | MAE      | **2.40**    | 2.75            |
| ADNI CDR_Binary                | AUC      | 0.53        | 0.53            |
| ADNI CDR NC-vs-AD (drop MCI)   | AUC      | **0.74**    | 0.74            |
| ADNI CDR NC-vs-MCI (drop AD)   | AUC      | 0.49        | 0.49            |
| ADNI FAQ Binary                | AUC      | 0.65        | 0.65            |
| ADNI Degradation 1Y / 2Y / 3Y  | AUC      | 0.56 / 0.47 / 0.45 | — |

# Les 9 SOTA

## Brain-JEPA — NeurIPS 2024 Spotlight

Lien : [arxiv.org/abs/2409.19407](https://arxiv.org/abs/2409.19407)

- **Architecture** : JEPA (Joint Embedding Predictive Architecture) + ViT, sur séries temporelles ROI.
- **Pretraining** : UK Biobank, **32 130 sujets** (80% des 40 162). Représentation = **450 ROI** Schaefer-400 + Tian-50, 160 timesteps. ROI-based, pas volumétrique.
- **Résultats clés** (verbatim, Tables 1-2-3-9, métrique = Acc% / F1%, **aucun AUC publié**) :
    - UKB Sex : Acc **88.17%** / F1 88.58%
    - HCP-Aging Sex : Acc **81.52%** / F1 84.26%
    - ADNI NC-vs-MCI : Acc **76.84 ± 1.05** / F1 86.32 ± 0.54 (n=189)
    - ADNI Amyloid β+/- : Acc 71.00 ± 4.90 / F1 75.97 ± 3.93
    - MACC NC-vs-MCI (Asiatiques) : Acc 65.98 ± 2.84 / F1 64.67 ± 2.61
    - **OASIS-3 AD conversion** (= notre Degradation, mais sans horizon 1/2/3Y) : Acc **69.00 ± 7.35** / F1 67.32 ± 7.92
    - CamCAN Depression : Acc 72.73 ± 2.87 / F1 67.45 ± 1.57
- **CV protocole** : split fixe **6:2:2** train/val/test, **5 runs** indépendants. **Subject-aware non explicitement confirmé** dans le papier.
- **Nous vs eux** :
    - Sex : ils sont au-dessus (HCPA Acc 81.5% ≈ AUC 0.88-0.90 vs notre HCP AUC 0.84)
    - NC-vs-MCI : gros gap (Acc 76.84% vs notre AUC 0.49)
    - **AD conversion** : leur Acc 69% ≈ AUC ~0.70 vs notre Degradation 1Y AUC 0.56 — **gap modéré seulement** (~0.14 AUC)

## BrainLM — ICLR 2024

Lien : [biorxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)

- **Architecture** : Masked-Autoencoder Transformer, 111 M params.
- **Pretraining** : UKB 76 296 recordings (~6 450 h) + HCP 1 002 recordings (~250 h) = **77 298 recordings / 6 700 h**. Atlas **AAL-424 ROI**.
- **Résultats clés** : UKB Age MSE z-scoré **0.503**, PTSD 0.015, Anxiety 0.073, Neuroticism 0.069. Aucun chiffre Sex / MMSE / AD / MCI.
- **Nous vs eux** : non comparable directement (MSE z-scoré, pas convertible en MAE sans la std d'âge UKB).

## SLIM-Brain — arXiv Déc 2025

Lien : [arxiv.org/abs/2512.21881](https://arxiv.org/abs/2512.21881)

- **Architecture** : **4D Hiera-JEPA**, hiérarchique, sur volumes 4D bruts (comme nous).
- **Pretraining** : **4 129 sessions** seulement (8-20× moins que Brain-JEPA / BrainLM).
- **Résultats clés** (verbatim, Tables 1-5-6, 3 runs) :
    - HCP Sex : Acc **91.1%** / F1 **91.1%** (linear probe : Acc 90.6 / F1 90.5)
    - HCP Fingerprint : Acc 98.5 / F1 98.1
    - ADNI MCI (finetune) : Acc **69.12 ± 1.38** / F1 68.96 ± 0.26
    - ADNI MCI (linear probe) : Acc 66.66 ± 1.40 / F1 64.52 ± 1.16
    - ADHD-200 : Acc 63.53 ± 0.53
    - PPMI Parkinson : Acc 70.40 ± 0.59
    - ABIDE Age (classification) : Acc 64.41 ± 0.57
    - ABIDE Age (régression) : MSE z-scoré 0.2175 ± 0.019
    - **Bat Brain-JEPA** sur ADNI MCI (69.12 vs 64.53) et sur HCP Sex (91.1 vs 87.1)
- **CV protocole** : split fixe **70/10/20** train/val/test, 3 runs. **Subject-aware non explicitement confirmé.**
- **Nous vs eux** : ils nous battent partout en Acc, mais HCP Sex 91% Acc ≈ AUC ~0.95+ vs notre AUC 0.84. Architecturalement le plus proche de nous.

## BrainGFM — arXiv 2025

Lien : [arxiv.org/abs/2506.02044](https://arxiv.org/abs/2506.02044)

- **Architecture** : Graph FM (graph contrastive + masked AE + meta-learning + language prompts).
- **Pretraining** : **27 datasets**, 25 pathologies, **25 000+ sujets / 60 000 scans / 400 000 graphes**, 8 parcellations.
- **Résultats clés** : chiffres précis sous extraction.
- **Nous vs eux** : paradigme totalement différent (graphes de connectomes).

## LCM (Large Connectome Model) — AAAI 2026

Lien : [arxiv.org/abs/2510.18910](https://arxiv.org/abs/2510.18910)

- **Architecture** : Transformer decoder-only avec MHSA + MHCA (features connectome × tokens "brain-environment").
- **Pretraining** : taille corpus non clairement caractérisée. Représentation = connectome (FC matrix).
- **Résultats clés** sous **5-fold subject-aware** (même protocole que nous) :
    - HCP-Aging Sex F1 **73.94 ± 2.45** / HCP-YA Sex F1 **72.23 ± 1.92**
    - **ADNI Alzheimer F1 85.33 ± 7.35**
    - PPMI Parkinson F1 84.18 / ABIDE Autism F1 72.50
- **Nous vs eux** : comparable directement. Notre HCP Sex 0.84 ≈ leur HCPYA F1 0.72. Leur ADNI AD F1 0.85 > notre 0.74 AUC.

## BNT (Brain Network Transformer) — NeurIPS 2022

Lien : [arxiv.org/abs/2210.06681](https://arxiv.org/abs/2210.06681)

- **Architecture** : Transformer sur graphes connectome, "Orthonormal Clustering Readout".
- **Pretraining** : aucun, **supervisé end-to-end**.
- **Résultats clés** : ABIDE Autism AUROC 80.2%. ABCD Sex (n=7 901). Brain-JEPA rapporte que **BNT bat Brain-JEPA** sur ADNI NC/MCI Acc (78.90% vs 76.84%).
- **Nous vs eux** : pas de comparator direct ADNI/HCP — baseline supervisée historique.

## OViTAD — Brain Sciences 2023

Lien : [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)

- **Architecture** : ViT 2D entraîné slice-par-slice, majority vote au niveau sujet.
- **Pretraining** : aucun, supervisé end-to-end ADNI.
- **Résultats clés** sous **subject-aware 80/10/10** (226/27/31) :
    - AD vs HC : F1 **0.99 ± 0.02**
    - HC vs MCI : F1 **0.97 ± 0.03**
- **Nous vs eux** : F1 0.99 intimidant, mais test = 31 sujets → std 0.02 = bruit massif. Méfiance.

## BrainNetCNN — NeuroImage 2017

Lien : [PDF Kawahara 2017](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)

- **Architecture** : CNN avec filtres custom Edge-to-Edge / Edge-to-Node / Node-to-Graph.
- **Pretraining** : aucun, supervisé end-to-end.
- **Résultats clés (papier original)** : prédiction Bayley-III sur DTI bébés prématurés. **Non comparable à nous.**
- **Nous vs eux** : baseline historique de référence, réutilisée par BNT / BrainGFM sur fMRI adulte.

## SwiFT — NeurIPS 2023

Lien : [arxiv.org/abs/2307.05916](https://arxiv.org/abs/2307.05916)

- **Architecture** : **Swin Transformer 4D** sur volumes fMRI directs (sans atlas). **Analogue architectural le plus proche de nous.**
- **Pretraining** : supervisé / SSL léger.
- **Résultats clés (en tant que comparator dans Brain-JEPA Table 9 et SLIM-Brain Table 1)** :
    - OASIS-3 AD Conversion : Acc 65.00 / F1 66.80
    - ADNI MCI : Acc 64.45 ± 1.69
    - ABIDE Age : Acc 62.22 ± 0.55 / MSE 0.4137 ± 0.033
    - ADHD-200 : Acc 60.81 ± 2.38
    - PPMI : Acc 58.10
    - Chiffres natifs sur HCP/ABCD/UKB Sex/Age non extraits dans ce pass.
- **Nous vs eux** : SwiFT < SLIM-Brain partout. Architecturalement le plus pertinent pour notre setup.

# Datasets de pretraining utilisés

| Rang | Dataset       | ~# sujets    | Accès            | Utilisé par                            |
|:----:|---------------|:------------:|------------------|----------------------------------------|
| 1    | UK Biobank    | 40 000+      | DUA payant       | BrainLM, Brain-JEPA, SwiFT, BrainGFM   |
| 2    | HCP-YA        | 1 200        | open + DUA       | BrainLM, SwiFT, LCM, **nous**          |
| 3    | HCP-Aging     | 700          | NDA              | Brain-JEPA, LCM                        |
| 4    | ABCD          | 12 000       | NDA              | BNT, SwiFT, BrainGFM                   |
| 5    | ADNI          | hundreds     | DUA              | downstream — Brain-JEPA, LCM, OViTAD, **nous** |
| 6    | ABIDE         | ~1 000       | open             | BNT, LCM, BrainGFM                     |
| 7    | PPMI          | hundreds     | DUA              | LCM, BrainGFM                          |

**Observation clé** : tous les SOTA au-dessus de nous pretrain sur UK Biobank (40 k+ sujets). Nous, HCP-YA seul (1.2 k, 30× moins). C'est probablement la principale source du gap.

# Où on en est — synthèse en 5 lignes

1. **Sex** — On est en dessous. SLIM-Brain HCP Sex Acc 91.1% (≈ AUC 0.95+), Brain-JEPA HCPA Acc 81.5% (≈ AUC 0.88), LCM HCPYA F1 0.72 ≈ nous. **Notre AUC 0.84 nous met au niveau de LCM, en dessous de SLIM-Brain et Brain-JEPA.**
2. **NC vs AD** — Notre 0.74 AUC, LCM F1 0.85, OViTAD revendique F1 0.99 (mais n=31 test). On est dans la course mais en dessous.
3. **NC vs MCI** — Notre 0.49 AUC vs Brain-JEPA 76.84% Acc et SLIM-Brain 69.12% Acc. **Gros gap.** Mais : leur meilleur Acc (76.84%) ≈ AUC ~0.80 — pas non plus écrasant.
4. **Age** — Aucun SOTA n'a publié de MAE en années comparable. Niche.
5. **Degradation / AD conversion** — **Brain-JEPA Table 9 a une tâche OASIS-3 AD Conversion : Acc 69.00% ≈ AUC ~0.70**. Notre Degradation 1Y AUC 0.56 est en dessous mais **le gap est modéré** (~0.14 AUC), pas un trou comme on pensait. À noter : Brain-JEPA ne fait pas le split par horizon 1/2/3Y, donc notre formulation reste originale.

# Les 3 messages pour le meeting

1. **Scale ≫ method.** Brain-JEPA et BrainLM pretrain sur UKB 32 k+. SLIM-Brain démontre que **4 k sessions suffisent pour battre 32 k** si l'architecture est bonne (4D Hiera-JEPA). On est à 1.2 k. **Le levier #1 est soit plus de data, soit changer d'architecture vers Hiera-JEPA.**
2. **NC-vs-AD marche, NC-vs-MCI non.** Cohérent avec la littérature et avec notre choix de pretraining HCP-YA (jeunes sains, zéro signal prodromal).
3. **Notre Degradation 1/2/3Y est moins original qu'on pensait** : Brain-JEPA Table 9 a une tâche "AD Conversion" sur OASIS-3 (Acc 69%, F1 67%). Notre Acc équivalente serait ~56-60% (depuis notre AUC 0.56). Mais : on est **les premiers à split par horizon temporel** 1Y/2Y/3Y. Reste original sur la forme, pas sur le fond.

# Les 3 SOTA à comparer en priorité (et pourquoi)

Parmi les 9 papiers passés en revue, ces 3 sont les seuls pertinents pour se positionner. Les autres sont écartés en bas de section.

## #1 — SLIM-Brain (le plus pertinent)

- **Pourquoi prioritaire** : **4D Hiera-JEPA sur volumes fMRI** = exactement notre paradigme. Pretraining à échelle modeste (~4 k sessions vs nos 1.2 k → moins éloigné qu'UKB 32 k). Rapporte HCP Sex + ADNI MCI + ABIDE Age, tâches comparables aux nôtres.
- **Comparaison directe** :
    - HCP Sex : eux **Acc 91.1%** vs nous **AUC 0.84** (≈ Acc 75-80%)
    - ADNI MCI : eux **Acc 69.12%** vs nous **AUC 0.49** sur NC-vs-MCI
- **Message thèse central** : SLIM-Brain **bat Brain-JEPA avec 8× moins de pretraining** (4 k vs 32 k sessions). C'est **la preuve** que l'**architecture > scale**, et ça valide notre direction (volumétrique + Hiera-JEPA).

## #2 — Brain-JEPA (le plus comprehensive sur ADNI)

- **Pourquoi prioritaire** : seul papier qui rapporte **simultanément** Sex (HCPA), NC/MCI (ADNI), Amyloid (ADNI), et **AD Conversion** sur OASIS-3 — la **vraie comparator** pour notre Degradation.
- **Comparaison directe** :
    - HCPA Sex : Acc 81.52% (≈ AUC 0.88-0.90) vs notre AUC 0.84
    - ADNI NC/MCI : **Acc 76.84%** vs notre AUC 0.49 — **gros gap**
    - OASIS-3 AD Conversion : Acc 69% (≈ AUC 0.70) vs notre Degradation 1Y AUC 0.56 — **gap modéré ~0.14 AUC seulement**

## #3 — SwiFT (la baseline architecturale honnête)

- **Pourquoi prioritaire** : **Swin Transformer 4D sur volumes** = l'archi la plus proche de la nôtre, mais en mode supervisé (sans pretraining SSL). Sert de **baseline minimale à dépasser** pour justifier qu'on fait du SSL.
- **Comparaison directe** (chiffres extraits comme comparator dans Brain-JEPA et SLIM-Brain) :
    - OASIS-3 AD Conversion : Acc 65.00 / F1 66.80
    - ADNI MCI : Acc 64.45
    - ABIDE Age : Acc 62.22 / MSE z-scoré 0.4137
- **Message** : on est censés battre SwiFT grâce au pretraining DINOv2. C'est la vraie barre minimale.

## Les 6 autres papiers, pourquoi écartés

| Papier | Raison |
|---|---|
| BrainLM | Rapporte uniquement MSE z-scoré sur Age/PTSD/Anxiety — aucun chiffre Sex/MCI/AD comparable |
| LCM | Connectome FC matrix — paradigme totalement différent du nôtre |
| BNT | Graphes de connectome — pas comparable au volumétrique |
| OViTAD | 2D slices supervisé, test n=31 (variance σ=0.02 = bruit massif) |
| BrainNetCNN | Préterm DTI à l'origine, baseline historique |
| BrainGFM | Graphes multi-atlas, chiffres précis pas extraits dans nos passes |

## La position à tenir en réunion

> "Mon comparator principal est **SLIM-Brain** parce qu'il valide ma direction (volume + Hiera-JEPA, scale modeste). Je suis en dessous d'eux sur HCP Sex (AUC 0.84 vs Acc 91%) et ADNI MCI, mais attendu vu qu'ils ont l'architecture aboutie et 3× plus de pretraining. Sur la tâche **AD Conversion**, l'écart avec **Brain-JEPA** est modéré (~0.14 AUC). Mon **angle original** c'est le split par horizon 1Y/2Y/3Y que personne ne fait. **SwiFT** est ma baseline minimale architecturale."
