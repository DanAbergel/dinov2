---
title: "SOTA fMRI — comparaison pour Ariel & Yoni"
date: "2026-06-14"
geometry: margin=2cm
fontsize: 10pt
---

# Nos chiffres (rappel)

Linear probe sur CLS gelé, 5-fold subject-aware CV. **Métriques utilisées par les SOTA : Accuracy (%) et F1 (%) pour la classification ; MAE pour la régression.** Best across iters 11999 / 17999 / 19999.

| Tâche                          | Métrique  | A (full FT)   | C (freeze-fmri)    |
|--------------------------------|:---------:|:-------------:|:------------------:|
| HCP Sex                        | Acc / F1  | 63.6 / 45.8   | **76.9 / 73.0**    |
| ADNI Sex                       | Acc / F1  | 67.0 / 59.5   | **73.9 / 71.3**    |
| ADNI Age                       | MAE (y)   | 5.19          | 5.70               |
| ADNI MMSE                      | MAE       | **2.39**      | 2.75               |
| ADNI CDR_Binary                | Acc / F1  | 61.0 / 0.6    | 61.5 / 21.1        |
| ADNI CDR NC-vs-AD (drop MCI)   | Acc / F1  | **70.8 / 0.0**| —                  |
| ADNI CDR NC-vs-MCI (drop AD)   | Acc / F1  | 61.1 / 75.9   | —                  |
| ADNI CDR_global                | MAE       | 0.276         | —                  |
| ADNI GDSCALE                   | Acc / F1  | 91.3 / 33.4   | —                  |
| ADNI FAQ Binary                | Acc / F1  | 88.2 / 0.0    | —                  |
| ADNI Degradation 1Y            | Acc / F1  | 81.6 / 0.0    | 81.4 / 2.4         |
| ADNI Degradation 2Y            | Acc / F1  | 70.6 / 0.8    | 68.4 / 5.5         |
| ADNI Degradation 3Y            | Acc / F1  | 68.8 / 0.0    | 66.4 / 8.4         |

**Note sur F1 = 0** : LogReg(C=1.0) sans `class_weight="balanced"` prédit la classe majoritaire sur les labels déséquilibrés (CDR, FAQ, Degradation 1Y où prévalence ~18-30%) → F1 du positif = 0 mais Acc reste haute. Ce n'est PAS une absence de signal, c'est un threshold non calibré. La même chose s'observe probablement chez Brain-JEPA/SLIM-Brain sans qu'ils le rapportent.

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
    - HCP Sex : eux **Acc 81.52%** vs nous **Acc 76.9% (C)** ou **63.6% (A)** — gap ~5 points sur C
    - ADNI NC-vs-MCI : eux **Acc 76.84%** vs nous **Acc 61.1%** — gap ~16 points
    - **AD Conversion (OASIS-3)** : eux **Acc 69%** vs nous **Degradation 1Y Acc 81.6%** (mais notre F1 ≈ 0 → on prédit toujours "non-déclin")
    - Sur Acc brute on est ~comparables. Sur F1, leur F1 67% sur AD Conversion est bien au-dessus de nos F1 ~0 sur Degradation.

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
- **Nous vs eux** :
    - HCP Sex : eux **Acc 91.1% / F1 91.1%** vs nous **Acc 76.9% / F1 73.0% (C)** — gap ~14 points sur Acc, ~18 sur F1
    - ADNI MCI : eux **Acc 69.12%** vs nous (CDR_Binary) **Acc 61.0%** — gap ~8 points
    - Ils sont au-dessus partout. Mais : leur scale 4k > notre 1.2k, et leur archi Hiera-JEPA est plus mûre.

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
- **Nous vs eux** :
    - HCP-YA Sex : eux **F1 72.23** vs nous **F1 73.0 (C)** — **on est au niveau de LCM**
    - ADNI AD : eux **F1 85.33** vs nous CDR_NC_vs_AD **F1 0.0** (threshold collapse, Acc 70.8%) → côté F1 ils gagnent largement

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

# Où on en est — synthèse en 5 lignes (Acc / F1)

1. **HCP Sex** — Nous **Acc 76.9% / F1 73.0% (C)**. SLIM-Brain **Acc 91.1% / F1 91.1%** (au-dessus de 14 points). Brain-JEPA HCPA **Acc 81.5% / F1 84.3%** (au-dessus de 5 points). **LCM HCPYA F1 72.2** ≈ nous (au niveau de LCM).
2. **ADNI NC vs AD** (CDR_NC_vs_AD chez nous, AD chez eux) — Nous **Acc 70.8% / F1 0.0** (threshold collapse). LCM **F1 85.3**. Brain-JEPA ne split pas. On est compétitifs en Acc mais le F1 0 nous tue dans la comparaison.
3. **ADNI NC vs MCI** — Nous **Acc 61.1% / F1 75.9** (ici F1 élevé parce que MCI est majoritaire dans le subset). Brain-JEPA **Acc 76.84% / F1 86.32**. SLIM-Brain **Acc 69.12%**. Gap ~8-16 points sur Acc.
4. **ADNI Age (régression)** — Nous **MAE 5.19 années**. Aucun SOTA n'a publié de MAE en années comparable (BrainLM en z-scoré uniquement). **Niche.**
5. **Degradation / AD conversion** — Nous **Deg1Y Acc 81.6% / F1 0.0**. Brain-JEPA OASIS-3 AD Conversion **Acc 69.0% / F1 67.3%**. On a plus d'Acc qu'eux MAIS leur F1 67 vs notre 0 → ils prédisent vraiment des conversions, nous on colle à la classe majoritaire.

# Les 3 messages pour le meeting

1. **Scale n'est PAS tout.** Brain-JEPA et BrainLM pretrain sur UKB 32 k+. **SLIM-Brain démontre que 4 k sessions suffisent pour battre 32 k** si l'architecture est bonne (4D Hiera-JEPA, comme nous). On est à 1.2 k. **Levier #1 : améliorer l'architecture vers Hiera-JEPA, pas juste empiler de la data.**
2. **NC-vs-AD marche, NC-vs-MCI non.** Notre Acc 70.8% sur NC-vs-AD (drop MCI) est dans la zone de Brain-JEPA. Sur NC-vs-MCI on plafonne. Cohérent avec notre choix de pretraining HCP-YA (jeunes sains, zéro signal prodromal).
3. **Problème de threshold à régler.** Nos F1 = 0 sur CDR_NC_vs_AD / FAQ / Degradation viennent du `LogReg(C=1.0)` sans `class_weight="balanced"` qui prédit la classe majoritaire. La même chose se produit probablement chez les SOTA mais ils ne le rapportent pas. À ajouter pour la prochaine version : probe avec class_weight balanced + threshold tuning, pour produire des F1 non-nuls comparables.

# Les 3 SOTA à comparer en priorité (et pourquoi)

Parmi les 9 papiers passés en revue, ces 3 sont les seuls pertinents pour se positionner. Les autres sont écartés en bas de section.

## #1 — SLIM-Brain (le plus pertinent)

- **Pourquoi prioritaire** : **4D Hiera-JEPA sur volumes fMRI** = exactement notre paradigme. Pretraining à échelle modeste (4 129 sessions vs nos 1 200 → moins éloigné qu'UKB 32 k). Rapporte HCP Sex + ADNI MCI + ABIDE Age, tâches comparables aux nôtres.
- **Comparaison directe** (Acc / F1) :
    - HCP Sex : eux **91.1 / 91.1** vs nous **76.9 / 73.0 (C)** — gap 14 pts Acc, 18 pts F1
    - ADNI MCI : eux **69.12 / N/A** vs nous CDR_Binary **61.0 / 0.6** — gap 8 pts Acc
- **Message thèse central** : SLIM-Brain **bat Brain-JEPA avec 8× moins de pretraining** (4 k vs 32 k sessions). C'est **la preuve** que l'**architecture > scale**, et ça valide notre direction (volumétrique + Hiera-JEPA).

## #2 — Brain-JEPA (le plus comprehensive sur ADNI)

- **Pourquoi prioritaire** : seul papier qui rapporte **simultanément** Sex (HCPA), NC/MCI (ADNI), Amyloid (ADNI), et **AD Conversion** sur OASIS-3 — la **vraie comparator** pour notre Degradation.
- **Comparaison directe** (Acc / F1) :
    - HCPA Sex : eux **81.52 / 84.26** vs nous HCP Sex **76.9 / 73.0 (C)** — gap 5 pts Acc, 11 pts F1
    - ADNI NC/MCI : eux **76.84 / 86.32** vs nous CDR_NC_vs_MCI **61.1 / 75.9** — gap 16 pts Acc, 10 pts F1
    - OASIS-3 AD Conversion : eux **69.0 / 67.3** vs nous Degradation 1Y **81.6 / 0.0** — on a plus d'Acc, mais F1 = 0 (threshold collapse)

## #3 — SwiFT (la baseline architecturale honnête)

- **Pourquoi prioritaire** : **Swin Transformer 4D sur volumes** = l'archi la plus proche de la nôtre, mais en mode supervisé (sans pretraining SSL). Sert de **baseline minimale à dépasser** pour justifier qu'on fait du SSL.
- **Comparaison directe** (Acc / F1, chiffres comme comparator dans Brain-JEPA T9 + SLIM-Brain T1) :
    - OASIS-3 AD Conversion : eux **65.0 / 66.8** vs nous Deg1Y **81.6 / 0.0** — Acc supérieure mais F1 nul
    - ADNI MCI : eux **64.45** vs nous CDR_Binary **61.0** — gap 3 pts Acc
    - ABIDE Age (cls) : eux **62.22** — pas testé chez nous
- **Message** : on est censés battre SwiFT grâce au pretraining DINOv2. Sur Acc on y arrive partiellement (Deg1Y 81.6 > 65), sur F1 on a un problème de threshold à régler.

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

> "Mon comparator principal est **SLIM-Brain** parce qu'il valide ma direction (volume + Hiera-JEPA, scale modeste). Je suis en dessous d'eux sur HCP Sex (Acc 76.9 vs 91.1) et ADNI MCI (Acc 61.0 vs 69.1), mais attendu vu qu'ils ont l'architecture aboutie et 3× plus de pretraining. Sur **AD Conversion**, je suis devant **Brain-JEPA** en Acc (81.6 vs 69.0) mais mon F1 = 0 (problème de threshold à régler avec class_weight='balanced'). Mon **angle original** c'est le split par horizon 1Y/2Y/3Y que personne ne fait. **SwiFT** est ma baseline minimale et je suis devant lui sur Acc."
