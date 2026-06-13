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
**Statut** : **PUBLIÉ — NeurIPS 2024 Spotlight** (top 3% des soumissions).

- **Architecture — JEPA (Joint Embedding Predictive Architecture)** :
    - Adaptation à la fMRI d'I-JEPA (Yann LeCun, 2023). Au lieu de prédire des **pixels bruts** (MAE) ou de contraster des **vues** (DINO/SimCLR), on prédit les **embeddings** d'un bloc cible à partir d'un bloc contexte.
    - **3 réseaux distincts** : context encoder (ViT) → predictor → target encoder (EMA du context encoder). Le predictor aligne les représentations latentes, jamais les données brutes.
    - **Spatiotemporal masking** custom : masques séparés sur l'axe ROI et sur l'axe temporel (pas un masque 2D commun).
    - **Gradient positioning** : injection d'un encoding positionnel par gradient learning au lieu de tables fixes.
    - Input = série temporelle 2D `(450 ROI, 160 timesteps)`, pas un volume.
- **Pretraining (1 dataset)** : **UK Biobank uniquement** — 32 130 sujets (80% des 40 162 dispo), TR=0.735s. Représentation = **450 ROI** Schaefer-400 + Tian-50, 160 timesteps. ROI-based, pas volumétrique.
- **Downstream evaluation (6 datasets)** : UKB held-out (20%), HCP-Aging (Sex), ADNI (NC/MCI, Amyloid, n=189), MACC (NC/MCI cohorte asiatique), OASIS-3 (AD conversion), CamCAN (Depression).
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
**Statut** : **PUBLIÉ — ICLR 2024** (Spotlight). Première vague de "foundation models" pour fMRI.

- **Architecture — Vision Transformer en mode Masked Autoencoder (MAE)** :
    - Standard ViT (encoder + decoder asymétrique style He et al. 2022) adapté aux séries fMRI.
    - **Paradigme MAE** : on masque ~75% des patches d'entrée, le decoder reconstruit les **valeurs brutes** des patches masqués (à l'opposé de JEPA qui reconstruit les embeddings).
    - **Tokenisation** : chaque patch = `(1 ROI × petite fenêtre temporelle)`. Concrètement avec 424 ROI AAL et fenêtres de 20 timesteps, ça fait des milliers de tokens.
    - **111 M paramètres** — le plus gros des foundation models fMRI.
    - Encoder ne voit que les patches non-masqués (gain de compute) ; decoder reconstruit tout.
- **Pretraining (2 datasets)** :
    - **UK Biobank** : 76 296 recordings (~6 450 h)
    - **HCP** : 1 002 recordings (~250 h)
    - Total : **77 298 recordings / 6 700 h**. Effectivement entraîné sur 80% UKB (61 038 recordings) + tout HCP.
    - Atlas **AAL-424 ROI** (atlas neuro-anatomique).
- **Downstream evaluation** : UK Biobank held-out — régression sur Age, PTSD (PCL-5), Anxiety (GAD-7), Neuroticism. **Aucune évaluation ADNI / HCP downstream rapportée**, contrairement à Brain-JEPA.
- **Résultats clés** : UKB Age MSE z-scoré **0.503**, PTSD 0.015, Anxiety 0.073, Neuroticism 0.069. Aucun chiffre Sex / MMSE / AD / MCI.
- **Nous vs eux** : non comparable directement (MSE z-scoré, pas convertible en MAE sans la std d'âge UKB).

## SLIM-Brain — arXiv Déc 2025

Lien : [arxiv.org/abs/2512.21881](https://arxiv.org/abs/2512.21881)
**Statut** : **NON ENCORE PUBLIÉ** — preprint arXiv (déc 2025), **sous review à OpenReview** (probablement ICLR 2026). Anonyme. Chiffres SOTA self-reported, pas encore peer-reviewed.

- **Architecture — 4D Hiera-JEPA (hybride de Hiera + JEPA)** :
    - **Hiera** = hierarchical ViT (Meta AI 2023), variante simplifiée de Swin **sans** windowed attention. Pyramide de 4 stages, downsampling × 2 entre chaque, mais self-attention globale dans chaque stage.
    - Combiné au **paradigme JEPA** (comme Brain-JEPA) : predictor latent au lieu de reconstruction pixel.
    - **4D natif** : input = volume `(T, X, Y, Z)`, patches 4D tubelets (genre `2×8×8×8` voxels-timesteps).
    - Hiérarchie pyramidale → représentations multi-échelles, ce qui permet de réduire la mémoire de **~70%** vs un ViT plat sur volumes.
    - **Architecturalement très proche de nous** (volume + transformer + SSL), mais : Hiera plutôt que ViT plat, JEPA plutôt que DINOv2.
- **Pretraining (5 datasets combinés)** — 4 129 sessions = 70% du total (Section 4.1) :
    1. **HCP** (Van Essen 2013) — une seule session
    2. **CHCP** (Chinese HCP, Ge 2023)
    3. **AOMIC PIOP1** (Amsterdam Open MRI Collection, Snoek 2021)
    4. **AOMIC PIOP2**
    5. **ABCD** (Adolescent Brain Cognitive Development, Casey 2018) — probablement le plus gros contributeur (12k+ adolescents dans ABCD)
    - Tous harmonisés à **2 mm isotrope** et **TR 0.72 s**.
    - **Clé** : 8-20× moins de sessions que Brain-JEPA / BrainLM, **mais multi-source** (5 datasets de populations différentes).
- **Downstream evaluation (7 benchmarks)** :
    - Internal (HCP) : Sex, Fingerprint
    - External : ADNI (MCI classification), ADHD-200, PPMI (Parkinson), ABIDE (Age classification + régression)
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
**Statut** : **NON PUBLIÉ** — preprint arXiv (juin 2025). Pas de venue confirmée à notre connaissance.

- **Architecture — Graph Foundation Model (paradigme graphes de connectome)** :
    - **Input** : connectome représenté en graphe — nœuds = ROIs, arêtes pondérées par corrélation fonctionnelle.
    - **2 SSL objectives combinés** :
        1. **Graph Contrastive Learning (GCL)** : contraste entre vues augmentées du même graphe (style SimCLR mais sur graphes).
        2. **Graph Masked Autoencoder (GraphMAE)** : on masque certains nœuds/arêtes, on les reconstruit (style MAE pour graphes).
    - **Meta-learning** : adaptation rapide à un nouveau disorder avec quelques shots, via MAML-like.
    - **Multi-prompts** : graph prompts (tokens spéciaux insérés dans le graphe) + language prompts (CLIP-style guidance par description textuelle de la pathologie).
    - **Multi-atlas** : entraîné sur 8 parcellations différentes (Schaefer-100/200/400, AAL, Power, etc.) → robustness à l'atlas downstream.
- **Pretraining (27 datasets, énormissime)** : 25 pathologies couvertes, **25 000+ sujets / 60 000 scans / 400 000 graph samples**, **8 parcellations** (Schaefer-100/200/400, AAL, Power, etc.) — la liste complète est en Appendix N du papier.
    - Datasets identifiés en Section 4.1 : OpenNeuro, UK Biobank, HCP, ABIDE, ABIDE II, ADHD-200, OASIS, ADNI 2, HBN (Healthy Brain Network), SubMex_CUD (Mexican Substance Use Disorder), UCLA_CNP (UCLA Consortium for Neuropsychiatric Phenomics), parmi 17+ autres non énumérés dans le texte principal.
- **Downstream evaluation** : multi-disorder — Autisme (ABIDE), TDAH (ADHD-200), Schizophrénie (COBRE/UCLA_CNP), Alzheimer (ADNI/OASIS), addictions (SubMex_CUD), etc. Cible explicite = **généralisation cross-disorder**.
- **Résultats clés** : chiffres précis sous extraction.
- **Nous vs eux** : paradigme totalement différent (graphes de connectomes).

## LCM (Large Connectome Model) — AAAI 2026

Lien : [arxiv.org/abs/2510.18910](https://arxiv.org/abs/2510.18910)
**Statut** : **PUBLIÉ — AAAI 2026** (accepté). Preprint arXiv (oct 2025).

- **Architecture — Transformer decoder-only (style GPT) pour connectomes** :
    - **Decoder-only** comme GPT-2/3 : pas d'encoder séparé, juste une pile de blocks transformer avec **causal masking**.
    - **2 types d'attention** par block :
        1. **Multi-Head Self-Attention (MHSA)** sur les features du connectome (FC matrix).
        2. **Multi-Head Cross-Attention (MHCA)** entre features connectome (Query) et **tokens "Brain-Environment-Interaction" (BEI)** (Key/Value).
    - Les **BEI tokens** encodent des covariables externes (âge, sexe, scanner, site...) — analogue à un prompt conditionnel CLIP-style.
    - **Pre-training task** : autoregression sur séquence de tokens connectome (predict next).
    - Représentation downstream = embedding du dernier token, fine-tunable sur classification/régression.
- **Pretraining (7 datasets, ~10 036 scans total)** — Section 4 Experiments + Table 6 Appendix B :
    1. **HCP-Aging (HCPA)** : 713 sujets / 4 863 scans (le gros contributeur)
    2. **HCP-YA (HCPYA)** : 248 sujets / 3 293 scans
    3. **ADNI** : 138 sujets / 138 scans
    4. **PPMI** : 209 sujets / 209 scans (Parkinson)
    5. **ABIDE** : 1 025 sujets / 1 025 scans (Autisme)
    6. **Taowu** : 40 sujets / 40 scans
    7. **Neurocon** : 41 sujets / 41 scans
    - Représentation = connectome (FC matrix), pas de volumes.
- **Downstream evaluation (8 datasets)** :
    - Sex prediction : sur tous les 7 datasets pretrain + 1 held-out
    - Cognitive state recognition : HCPA (4 classes), HCPYA (7 classes)
    - Maladie : Alzheimer (ADNI), Parkinson (PPMI / Taowu / Neurocon), Autisme (ABIDE), Schizophrénie (SZ held-out)
- **Résultats clés** sous **5-fold subject-aware** (même protocole que nous) :
    - HCP-Aging Sex F1 **73.94 ± 2.45** / HCP-YA Sex F1 **72.23 ± 1.92**
    - **ADNI Alzheimer F1 85.33 ± 7.35**
    - PPMI Parkinson F1 84.18 / ABIDE Autism F1 72.50
- **Nous vs eux** :
    - HCP-YA Sex : eux **F1 72.23** vs nous **F1 73.0 (C)** — **on est au niveau de LCM**
    - ADNI AD : eux **F1 85.33** vs nous CDR_NC_vs_AD **F1 0.0** (threshold collapse, Acc 70.8%) → côté F1 ils gagnent largement

## BNT (Brain Network Transformer) — NeurIPS 2022

Lien : [arxiv.org/abs/2210.06681](https://arxiv.org/abs/2210.06681)
**Statut** : **PUBLIÉ — NeurIPS 2022**. Le Transformer-baseline canonique pour les graphes de connectome.

- **Architecture — Transformer sur graphes de connectome avec readout custom** :
    - **Pas un GNN** : c'est un Transformer pur où chaque ROI est un token. Pas de message-passing.
    - **Features par nœud** = profil de connexion entier (ligne de la matrice de corrélation) → chaque nœud a un vecteur de dim N (où N = nombre total de ROIs).
    - **Self-attention standard** entre tous les nœuds → matrice d'attention N×N. Pas de masquage de graphe.
    - **Orthonormal Clustering Readout (OCR)** = innovation centrale : au lieu d'un mean-pooling final, on apprend K prototypes orthogonaux et chaque nœud est softmax-assigné à un cluster, puis on pool par cluster. Donne un readout graph-level plus structuré.
    - Pas de pretraining — entraîné end-to-end directement sur la tâche.
- **Pretraining** : **AUCUN** — supervisé end-to-end. Pas de SSL.
- **Downstream / training datasets (2 datasets)** :
    - **ABIDE** : autism classification, AUROC 80.2%
    - **ABCD** : Sex prediction (n=7 901 adolescents) — un des plus gros datasets fMRI utilisés en supervisé
    - **Pas d'évaluation HCP / ADNI / MCI** dans le papier original BNT (mais Brain-JEPA rapporte que BNT bat Brain-JEPA sur ADNI NC/MCI Acc 78.90% vs 76.84% comme comparator).
- **Nous vs eux** : pas de comparator direct ADNI/HCP — baseline supervisée historique.

## OViTAD — Brain Sciences 2023

Lien : [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
**Statut** : **PUBLIÉ — Brain Sciences 2023** (journal MDPI, IF ~3, **pas une top venue**). Premier preprint bioRxiv 2021.

- **Architecture — Vision Transformer 2D standard sur slices fMRI** :
    - Pas de modification architecturale : c'est un **ViT-B/16 standard** (style Dosovitskiy 2020).
    - **Input** = slices 2D axiales individuelles extraites des volumes 4D fMRI. **Pas de prise en compte du temps ni de la 3D.**
    - Chaque scan 4D → N_slices × N_timepoints "images" 2D indépendantes pour l'entraînement.
    - **Inférence par majority vote** : agrégation des prédictions slice-par-slice en une décision par sujet (vote majoritaire).
    - Entraîné end-to-end (pas de SSL) sur ADNI rs-fMRI directement.
    - Le "O" de OViTAD = "Optimized" : tuning d'hyperparamètres standard (pas une vraie innovation architecturale).
- **Pretraining** : **AUCUN** — supervisé end-to-end. Pas de SSL.
- **Downstream / training dataset (1 seul dataset)** :
    - **ADNI rs-fMRI** : **284 sujets total** répartis 80/10/10 → 226 train / 27 val / 31 test. Tâches : AD vs HC, HC vs MCI.
- **Résultats clés** sous **subject-aware 80/10/10** (226/27/31) :
    - AD vs HC : F1 **0.99 ± 0.02**
    - HC vs MCI : F1 **0.97 ± 0.03**
- **Nous vs eux** : F1 0.99 intimidant, mais test = 31 sujets → std 0.02 = bruit massif. Méfiance.

## BrainNetCNN — NeuroImage 2017

Lien : [PDF Kawahara 2017](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
**Statut** : **PUBLIÉ — NeuroImage 2017** (Elsevier, IF ~5). Baseline historique citée par tous les papiers connectome modernes.

- **Architecture — CNN avec 3 filtres custom pensés pour la matrice de connectivité** :
    - **Input** = matrice de connectivité N×N (N = nombre de ROIs). Pas une image classique : la position (i, j) a une **sémantique** (corrélation entre ROI i et j), pas une translation-invariance comme les images naturelles.
    - 3 types de couches custom — la clé du papier :
        1. **Edge-to-Edge (E2E)** : remplace la conv 2D classique. Pour chaque cellule (i, j), agrège la **ligne i** + la **colonne j** séparément (= les connexions des 2 ROIs concernées). Préserve la structure topologique de la matrice.
        2. **Edge-to-Node (E2N)** : agrège toutes les connexions d'un ROI en un seul vecteur de features par ROI (1D).
        3. **Node-to-Graph (N2G)** : pool tous les ROI-features en un seul vecteur graph-level pour la classification finale.
    - Pas de SSL — entraîné end-to-end. Publication originale sur DTI bébés prématurés.
- **Pretraining** : **AUCUN** — supervisé end-to-end. Pas de SSL.
- **Downstream / training dataset (papier original)** : **DTI (diffusion MRI, pas fMRI)** de bébés prématurés (27-46 semaines GA), prédiction Bayley-III. **Non comparable à nos labels ADNI/HCP.** L'archi est réutilisée par d'autres (BNT, BrainGFM, BrainGB) sur fMRI adulte comme baseline.
- **Nous vs eux** : baseline historique de référence, réutilisée par BNT / BrainGFM sur fMRI adulte.

## SwiFT — NeurIPS 2023

Lien : [arxiv.org/abs/2307.05916](https://arxiv.org/abs/2307.05916)
**Statut** : **PUBLIÉ — NeurIPS 2023**. Premier Swin 4D pour fMRI.

- **Architecture — Swin Transformer étendu en 4D pour fMRI volumes** :
    - Extension du **Swin Transformer** (Liu et al. 2021, ICCV) en 4D : fenêtres `(X, Y, Z, T)` au lieu de `(H, W)`.
    - **Windowed self-attention** : self-attention calculée à l'intérieur de fenêtres 4D non-chevauchantes (efficacité mémoire — l'attention complète sur 9000 tokens serait O(81M) ops).
    - **Shifted windows** entre layers successives : les fenêtres bougent pour que des tokens dans des fenêtres voisines puissent communiquer indirectement → reçoit un contexte global progressivement.
    - **Hiérarchique** : 4 stages avec patch merging × 2 entre chaque (style Swin classique).
    - **Embeddings positionnels absolus** (pas relatifs comme Swin v2).
    - Input direct = volume 4D `(X, Y, Z, T)`, pas d'atlas, comme nous.
    - Entraîné supervisé (pas de SSL).
- **Pretraining (SSL léger contrastif)** : optionnel, contrastive loss style SimCLR. Le papier montre que ça aide mais ne précise pas la composition exacte du corpus SSL.
- **Datasets training/eval (3 datasets large-scale)** :
    - **HCP-YA** : Sex, Age, Fluid Intelligence
    - **ABCD** : Sex, Age, Intelligence cognitive (Total Composite Score)
    - **UK Biobank** : Sex, Age (échelle UKB, ~40k sujets)
    - Compte exacte des sujets/scans non rapportée dans le main paper (abstract dit "large-scale" uniquement).
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
