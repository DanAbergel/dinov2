---
title: "fMRI prediction models — SOTA review"
date: "2026-06-14"
geometry: margin=2cm
fontsize: 10pt
---

# SOTA models

## 1. Brain-JEPA — NeurIPS 2024 Spotlight

**arXiv**: [2409.19407](https://arxiv.org/abs/2409.19407)
**Status**: Published — NeurIPS 2024 Spotlight (top 3% of submissions).
**Architecture**: JEPA (Joint Embedding Predictive Architecture) + ViT on **ROI time-series** (450 ROI × 160 timesteps). The predictor reconstructs the **embeddings** of target blocks instead of raw voxels (LeCun's paradigm, adapted from I-JEPA).
**Pretraining dataset**: UK Biobank — 32,130 subjects (80% of 40,162 available), TR 0.735s.

| Evaluation dataset | Task                     | Acc       | F1        |
|--------------------|--------------------------|:---------:|:---------:|
| UK Biobank         | Sex                      | **88.17** | **88.58** |
| HCP-Aging          | Sex                      | **81.52** | **84.26** |
| ADNI (n=189)       | NC vs MCI                | **76.84** | **86.32** |
| ADNI               | Amyloid β+/-             | 71.00     | 75.97     |
| MACC               | NC vs MCI (Asian cohort) | 65.98     | 64.67     |
| OASIS-3            | AD Conversion            | 69.00     | 67.32     |
| CamCAN             | Depression               | 72.73     | 67.45     |

*Downstream protocol: fixed 6:2:2 train/val/test split across all datasets, results averaged over 5 independent runs. Subject-awareness not explicitly stated.*

---

## 2. BrainLM — ICLR 2024 Spotlight

**arXiv**: [biorxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)
**Status**: Published — ICLR 2024 Spotlight.
**Architecture**: Vision Transformer in Masked Autoencoder (MAE) mode on ROI time-series. Masks 75% of patches, reconstructs **raw values**. 111M parameters.
**Pretraining datasets**: UK Biobank (76,296 recordings, ~6,450 hours) + HCP (1,002 recordings, ~250 hours) = 77,298 recordings / 6,700 hours. Atlas: AAL-424 ROI.

| Evaluation dataset | Task                     | MSE (z-scored)        |
|--------------------|--------------------------|:---------------------:|
| UK Biobank         | Age (regression)         | **0.503 ± 0.021**     |
| UK Biobank         | PTSD (PCL-5)             | 0.015 ± 0.0003        |
| UK Biobank         | Anxiety (GAD-7)          | 0.073 ± 0.003         |
| UK Biobank         | Neuroticism              | 0.069 ± 0.004         |

*No Sex / MCI / AD / MMSE numbers reported. Only z-scored MSE regression on UK Biobank held-out.*

---

## 3. SLIM-Brain — preprint (Dec 2025, under OpenReview review)

**arXiv**: [2512.21881](https://arxiv.org/abs/2512.21881)
**Status**: Not yet published — preprint, under OpenReview review (anonymous submission). Self-reported SOTA, not peer-reviewed.
**Architecture**: **4D Hiera-JEPA** — Hiera (hierarchical pyramidal ViT, multi-stage) combined with the JEPA paradigm. Input = **4D volumes `(T, X, Y, Z)`**. 4D tubelets, global attention within each stage.
**Pretraining datasets**: HCP + CHCP + AOMIC PIOP1 + AOMIC PIOP2 + ABCD = **4,129 sessions** combined (70% of total). Harmonized to 2mm isotropic / TR 0.72s.

| Evaluation dataset | Task                     | Acc       | F1 / other  |
|--------------------|--------------------------|:---------:|:-----------:|
| HCP                | Sex                      | **91.1**  | F1 **91.1** |
| HCP                | Fingerprint              | 98.5      | F1 98.1     |
| ADNI               | MCI classification       | **69.12** | F1 68.96    |
| ADHD-200           | ADHD                     | 63.53     | —           |
| PPMI               | Parkinson                | 70.40     | —           |
| ABIDE              | Age (classification)     | 64.41     | —           |
| ABIDE              | Age (regression)         | —         | MSE 0.2175  |

*Beats Brain-JEPA on HCP Sex (87.1) and ADNI MCI (64.53). Downstream protocol: 70/10/20 train/val/test split, 3 independent runs. Subject-awareness not explicitly stated.*

---

## 4. BrainGFM — preprint (June 2025)

**arXiv**: [2506.02044](https://arxiv.org/abs/2506.02044)
**Status**: Not published — preprint, no confirmed venue.
**Architecture**: **Graph Foundation Model** on connectomes. Combines **Graph Contrastive Learning + Graph Masked Autoencoder** + meta-learning + graph prompts + language prompts. Multi-atlas (8 parcellations).
**Pretraining datasets**: 27 datasets covering 25 disorders, 25,000+ subjects / 60,000 scans / 400,000 graph samples. Identified examples: OpenNeuro, UK Biobank, HCP, ABIDE, ABIDE II, ADHD-200, OASIS, ADNI 2, HBN, SubMex_CUD, UCLA_CNP (full list in Appendix N).

| Evaluation dataset | Task                | AUC      | Acc      |
|--------------------|---------------------|:--------:|:--------:|
| ADHD-200           | ADHD                | 70.6     | 72.2     |
| ABIDE II           | Autism (ASD)        | 71.2     | 73.5     |
| **ADNI 2**         | **Alzheimer (AD)**  | **80.3** | **85.1** |
| HBN                | Major Depression    | 83.6     | 85.5     |
| HBN                | Anxiety             | 85.2     | 86.3     |
| HBN                | OCD                 | 80.4     | 85.8     |
| HBN                | PTSD                | 83.2     | 86.3     |
| SubMex_CUD         | Cocaine Use         | 71.1     | 74.6     |
| UCLA_CNP           | Schizophrenia       | 84.2     | 86.7     |
| UCLA_CNP           | Bipolar             | 73.5     | 76.3     |

*No Sex / Age / Parkinson results reported. Schaefer-100 atlas used.*

---

## 5. LCM (Large Connectome Model) — AAAI 2026

**arXiv**: [2510.18910](https://arxiv.org/abs/2510.18910)
**Status**: Published — AAAI 2026 (accepted).
**Architecture**: **Decoder-only Transformer** (GPT-style) on connectomes (FC matrix). MHSA + MHCA between connectome features and "brain-environment-interaction" (BEI) tokens (encoding covariates: sex, age, scanner, site).
**Pretraining datasets**: HCPA (713 subj / 4,863 scans) + HCPYA (248 / 3,293) + ADNI (138 / 138) + PPMI (209 / 209) + ABIDE (1,025 / 1,025) + Taowu (40 / 40) + Neurocon (41 / 41) = ~10,036 scans / 7 datasets.

| Evaluation dataset | Task             | F1                |
|--------------------|------------------|:-----------------:|
| HCP-Aging          | Sex              | **73.94 ± 2.45**  |
| HCP-YA             | Sex              | **72.23 ± 1.92**  |
| ABIDE              | Sex              | 87.34 ± 4.48      |
| **ADNI**           | **Alzheimer**    | **85.33 ± 7.35**  |
| PPMI               | Parkinson        | 84.18 ± 11.63     |
| ABIDE              | Autism           | 72.50 ± 1.91      |

*Downstream protocol: subject-aware 5-fold CV for HCPA/HCPYA/ADNI, 10-fold for others. Pretraining and finetuning always from the same CV fold's training set to prevent leakage.*

---

## 6. BNT (Brain Network Transformer) — NeurIPS 2022

**arXiv**: [2210.06681](https://arxiv.org/abs/2210.06681)
**Status**: Published — NeurIPS 2022.
**Architecture**: Pure Transformer on connectome graphs (not a GNN). Each ROI = one token, node features = connection profile. Key innovation = **Orthonormal Clustering Readout** (pooling via K learned orthogonal prototypes).
**Pretraining**: None — supervised end-to-end.

| Evaluation dataset | Task           | Metric    | Score        |
|--------------------|----------------|-----------|:------------:|
| ABIDE              | Autism         | AUROC     | **80.2%**    |
| ABCD               | Sex (n=7,901)  | —         | not extracted |
| ADNI               | NC vs MCI      | Acc       | **78.90**    |

*BNT beats Brain-JEPA on ADNI NC/MCI (78.90 vs 76.84) — reported as comparator in Brain-JEPA paper.*

---

## 7. OViTAD — Brain Sciences 2023

**arXiv**: [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
**Status**: Published — Brain Sciences 2023 (MDPI journal, IF ~3, not a top venue).
**Architecture**: Standard **ViT-B/16** on individual **2D fMRI slices** (no 3D, no time). Subject-level inference via **majority vote**. No architectural innovation — only hyperparameter tuning.
**Pretraining**: None — supervised end-to-end on ADNI.

| Evaluation dataset | Task        | F1               |
|--------------------|-------------|:----------------:|
| ADNI (n=284)       | AD vs HC    | **0.99 ± 0.02**  |
| ADNI (n=284)       | HC vs MCI   | **0.97 ± 0.03**  |

*Test set = only 31 subjects. σ=0.02 on n=31 is noise-dominated. Subject-aware 80/10/10 split (226 train / 27 val / 31 test).*

---

## 8. BrainNetCNN — NeuroImage 2017

**arXiv**: [Kawahara 2017 PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
**Status**: Published — NeuroImage 2017 (Elsevier, IF ~5). Historical baseline reused by modern connectome papers.
**Architecture**: CNN with **3 custom topology-aware filters** on connectivity matrix — Edge-to-Edge (E2E, aggregates row+column per cell), Edge-to-Node (E2N, aggregates per ROI), Node-to-Graph (N2G, global pooling).
**Pretraining**: None — supervised end-to-end.

| Evaluation dataset       | Task              | Score              |
|--------------------------|-------------------|:------------------:|
| Preterm DTI (original)   | Bayley-III scores | **not comparable** |

*Original paper used DTI (diffusion MRI, not fMRI) on preterm infants. Architecture reused as supervised baseline by BNT / BrainGFM / BrainGB on adult fMRI.*

---

## 9. SwiFT — NeurIPS 2023

**arXiv**: [2307.05916](https://arxiv.org/abs/2307.05916)
**Status**: Published — NeurIPS 2023.
**Architecture**: **4D Swin Transformer** on raw fMRI volumes (no atlas). 4D windows `(X, Y, Z, T)` + windowed self-attention + shifted windows between layers + hierarchical (4 stages). Closest volumetric Transformer paradigm.
**Pretraining**: Light contrastive SSL (composition not specified in main paper).

| Evaluation dataset | Task              | Acc       | Other        |
|--------------------|-------------------|:---------:|:------------:|
| OASIS-3            | AD Conversion     | 65.00     | F1 66.80     |
| ADNI               | MCI               | 64.45     | —            |
| ABIDE              | Age (cls)         | 62.22     | MSE 0.4137   |
| ADHD-200           | ADHD              | 60.81     | —            |
| PPMI               | Parkinson         | 58.10     | —            |
| HCP / ABCD / UKB   | Sex, Age          | —         | not extracted |

*Numbers reported as comparator within Brain-JEPA Table 9 and SLIM-Brain Table 1 — not directly extracted from the SwiFT paper itself.*

---

# Datasets — dimensions and access

| Dataset       | # subjects | TR (s) | Typical T | Access mode                              | Used in pretraining by | Used in evaluation by |
|---------------|:----------:|:------:|:---------:|------------------------------------------|------------------------|------------------------|
| UK Biobank    | 40,000+    | 0.735  | 490       | Paid DUA (~£500-1000), 2-6 months        | Brain-JEPA, BrainLM, BrainGFM | Brain-JEPA, BrainLM, SwiFT |
| HCP-YA        | 1,200      | 0.72   | 1,200     | Open with DUA, fast                      | BrainLM, SwiFT, LCM    | LCM, SLIM-Brain (Sex)  |
| HCP-Aging     | 700        | 0.8    | ~470      | NDA controlled, 1-3 months               | Brain-JEPA (eval only), LCM | Brain-JEPA, LCM     |
| CHCP          | ~370       | 0.72   | varies    | Chinese open access                      | SLIM-Brain             | —                      |
| AOMIC PIOP1   | 216        | 2.0    | ~480      | Open access                              | SLIM-Brain             | —                      |
| AOMIC PIOP2   | 226        | 2.0    | ~480      | Open access                              | SLIM-Brain             | —                      |
| ABCD          | 11,800+    | 0.8    | ~375      | NDA controlled, 1-3 months               | SLIM-Brain, SwiFT, BrainGFM | SwiFT             |
| ADNI / ADNI 2 | ~1,000     | 3.0    | ~140      | DUA, 1-2 weeks                           | LCM (138 subj only)    | Brain-JEPA, BrainGFM, LCM, SLIM-Brain, OViTAD |
| ABIDE I       | 1,112      | 1.5-3  | 76-300    | Open (NITRC / Preprocessed Connectomes Project) | BrainGFM        | BNT, Brain-JEPA, BrainGFM, SLIM-Brain, LCM |
| ABIDE II      | 1,114      | varies | varies    | Open                                     | BrainGFM               | BrainGFM               |
| ADHD-200      | 776        | 1.5-2.5| 70-280    | Open (NITRC)                             | BrainGFM               | BrainGFM, SLIM-Brain   |
| PPMI          | ~500       | 2.4    | ~200      | DUA, 1-2 weeks                           | LCM                    | LCM, BrainGFM, SLIM-Brain |
| HBN           | 3,000+     | 0.8    | varies    | Open (AWS S3)                            | BrainGFM               | BrainGFM               |
| OASIS-3       | 1,098      | 2.2    | ~180      | Registration + proposal, 1-7 days        | —                      | Brain-JEPA (AD Conversion) |
| MACC          | restricted | varies | varies    | Restricted (Asian collaborators only)    | —                      | Brain-JEPA (NC/MCI Asian) |
| CamCAN        | 700        | 1.97   | 261       | Open with registration                   | —                      | Brain-JEPA (Depression) |
| SubMex_CUD    | restricted | varies | varies    | Restricted                               | BrainGFM               | BrainGFM (Cocaine)     |
| UCLA_CNP      | 272        | 2.0    | 152       | Open (OpenNeuro)                         | BrainGFM               | BrainGFM (Schizo, Bipolar) |
| Taowu         | 40         | 2.0    | ~120      | Open (OpenNeuro)                         | LCM                    | LCM (Parkinson)        |
| Neurocon      | 41         | 2.0    | ~140      | Open (OpenNeuro)                         | LCM                    | LCM (Parkinson)        |

**Access categories**:
- **Open access** (immediate download via Python libraries like `nilearn`): ABIDE I/II, ADHD-200, AOMIC PIOP1/PIOP2, CHCP, HBN, UCLA_CNP, Taowu, Neurocon
- **Registration + short proposal** (1-7 days): OASIS-3, CamCAN
- **DUA, fast** (1-2 weeks): ADNI, PPMI, HCP-YA
- **NDA controlled** (1-3 months): HCP-Aging, ABCD
- **Paid DUA, slow** (2-6 months): UK Biobank (~£500-1000)
- **Restricted to specific collaborators**: MACC, SubMex_CUD

---

# Cross-reference tables

## Datasets used for pretraining vs evaluation

| Model        | Pretraining datasets                                   | # pretrain sessions | Evaluation datasets                                 |
|--------------|---------------------------------------------------------|:-------------------:|------------------------------------------------------|
| Brain-JEPA   | UK Biobank                                              | ~32,130 subjects    | UKB, HCP-Aging, ADNI, MACC, OASIS-3, CamCAN          |
| BrainLM      | UK Biobank + HCP                                        | 77,298 recordings   | UK Biobank only                                      |
| SLIM-Brain   | HCP + CHCP + AOMIC PIOP1/PIOP2 + ABCD                   | 4,129 sessions      | HCP, ADNI, ADHD-200, PPMI, ABIDE                     |
| BrainGFM     | 27 datasets (HCP, UKB, ABIDE, ADHD-200, OASIS, ADNI 2, HBN, SubMex_CUD, UCLA_CNP, +18 in App N) | ~60,000 scans | ADNI 2, ABIDE II, ADHD-200, HBN, SubMex_CUD, UCLA_CNP |
| LCM          | HCPA + HCPYA + ADNI + PPMI + ABIDE + Taowu + Neurocon   | ~10,036 scans       | HCPA, HCPYA, ABIDE, ADNI, PPMI                       |
| BNT          | None (end-to-end supervised)                            | —                   | ABIDE, ABCD                                          |
| OViTAD       | None (end-to-end supervised)                            | —                   | ADNI (284 subjects)                                  |
| BrainNetCNN  | None (end-to-end supervised)                            | —                   | Preterm DTI (original paper)                         |
| SwiFT        | Light contrastive SSL (corpus unspecified)              | —                   | HCP, ABCD, UK Biobank                                |

## Labels used per model

| Model        | Label types                                                                              |
|--------------|------------------------------------------------------------------------------------------|
| Brain-JEPA   | DX (NL/MCI clinical diagnosis), Amyloid β +/-, Sex, Depression                            |
| BrainLM      | Age (regression), PTSD, Anxiety, Neuroticism (all z-scored MSE)                          |
| SLIM-Brain   | DX (MCI classification), Sex, Fingerprint, ADHD diagnosis, Parkinson, Age                |
| BrainGFM     | 10 disorders: ADHD, ASD, AD (DX), MDD, Anxiety, OCD, PTSD, Cocaine, Schizo, Bipolar       |
| LCM          | Sex, DX (AD, PD, ASD, SZ)                                                                |
| BNT          | DX (Autism), Sex                                                                          |
| OViTAD       | DX (AD vs HC, HC vs MCI)                                                                  |
| BrainNetCNN  | Bayley-III scores (preterm)                                                              |
| SwiFT        | Sex, Age, Cognitive intelligence                                                          |

---

# Notes on metric comparability

- **Most SOTA papers report Accuracy (%) and F1 (%)**, not AUC. Direct AUC-to-Acc conversion is approximate.
- **Brain-JEPA / BrainGFM / LCM use DX clinical diagnosis labels** (NL / MCI / AD from ADNI metadata). Direct comparison to CDR-based binarisations requires label alignment.
- **No SOTA paper reports horizon-specific cognitive decline forecasting** (1Y / 2Y / 3Y splits). The closest comparator is Brain-JEPA's OASIS-3 AD Conversion (single horizon, Acc 69.00 / F1 67.32).
- **Subject-level splits**: Brain-JEPA (6:2:2), SLIM-Brain (70:10:20), LCM (subject-aware 5/10-fold), OViTAD (80:10:10 at participant level). Subject-awareness explicitly confirmed only for LCM and OViTAD.

---

# Sources

1. **Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking** — Dong & Li et al., NeurIPS 2024 Spotlight — [arxiv.org/abs/2409.19407](https://arxiv.org/abs/2409.19407)
2. **BrainLM: A foundation model for brain activity recordings** — Ortega Caro et al., ICLR 2024 — [biorxiv 10.1101/2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)
3. **SLIM-Brain: A Data- and Training-Efficient Foundation Model for fMRI Data Analysis** — Anonymous, arXiv Dec 2025 — [arxiv.org/abs/2512.21881](https://arxiv.org/abs/2512.21881)
4. **Brain Graph Foundation Model (BrainGFM): Pre-Training and Prompt-Tuning across Broad Atlases and Disorders** — arXiv 2025 — [arxiv.org/abs/2506.02044](https://arxiv.org/abs/2506.02044)
5. **Large Connectome Model (LCM): A Decoder-Only Foundation Model for fMRI** — AAAI 2026 — [arxiv.org/abs/2510.18910](https://arxiv.org/abs/2510.18910)
6. **Brain Network Transformer (BNT)** — Kan, Cui et al., NeurIPS 2022 — [arxiv.org/abs/2210.06681](https://arxiv.org/abs/2210.06681)
7. **OViTAD: Optimized Vision Transformer for the diagnosis of Alzheimer's disease** — Sarraf et al., Brain Sciences 2023 — [biorxiv 10.1101/2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
8. **BrainNetCNN: Convolutional neural networks for brain networks** — Kawahara et al., NeuroImage 2017 — [PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
9. **SwiFT: Swin 4D fMRI Transformer** — Kim et al., NeurIPS 2023 — [arxiv.org/abs/2307.05916](https://arxiv.org/abs/2307.05916)
