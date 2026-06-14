---
title: "fMRI prediction models — SOTA review"
date: "2026-06-14"
geometry: margin=2cm
fontsize: 10pt
---

# 1. Brain-JEPA

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — NeurIPS 2024 Spotlight       |
| **arXiv**    | [2409.19407](https://arxiv.org/abs/2409.19407) |

### Architecture

JEPA (Joint Embedding Predictive Architecture) + ViT on **ROI time-series** (450 ROI × 160 timesteps). The predictor reconstructs the **embeddings** of target blocks instead of raw voxels (LeCun's I-JEPA paradigm adapted to fMRI).

### Pretraining dataset

- **UK Biobank** — 32,130 subjects (80% of 40,162 available), TR 0.735s

### Downstream evaluation datasets

- UK Biobank held-out (Sex)
- HCP-Aging (Sex)
- ADNI (n=189) — NC vs MCI, Amyloid β+/-
- MACC — NC vs MCI Asian cohort
- OASIS-3 — AD Conversion
- CamCAN — Depression

### Results (Accuracy / F1, %)

| Dataset       | Task                | Acc       | F1        |
|---------------|---------------------|:---------:|:---------:|
| UK Biobank    | Sex                 | **88.17** | **88.58** |
| HCP-Aging     | Sex                 | **81.52** | **84.26** |
| ADNI (n=189)  | NC vs MCI           | **76.84** | **86.32** |
| ADNI          | Amyloid β+/-        | 71.00     | 75.97     |
| MACC          | NC vs MCI           | 65.98     | 64.67     |
| OASIS-3       | AD Conversion       | 69.00     | 67.32     |
| CamCAN        | Depression          | 72.73     | 67.45     |

### Protocol

Fixed 6:2:2 train/val/test split across all downstream datasets. Results averaged over 5 independent runs. Subject-awareness not explicitly stated.

\newpage

# 2. BrainLM

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — ICLR 2024 Spotlight          |
| **arXiv**    | [biorxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1) |

### Architecture

Vision Transformer as Masked Autoencoder (MAE). Masks 75% of input patches, reconstructs **raw values**. 111M parameters.

### Pretraining datasets

- **UK Biobank** — 76,296 recordings (~6,450 hours)
- **HCP** — 1,002 recordings (~250 hours)
- **Total** — 77,298 recordings / 6,700 hours
- Atlas: AAL-424 ROI

### Downstream evaluation datasets

- UK Biobank held-out only — 4 regression tasks

### Results (MSE on z-scored targets)

| Dataset    | Task              | MSE (z-scored)    |
|------------|-------------------|:-----------------:|
| UK Biobank | Age               | **0.503 ± 0.021** |
| UK Biobank | PTSD (PCL-5)      | 0.015 ± 0.0003    |
| UK Biobank | Anxiety (GAD-7)   | 0.073 ± 0.003     |
| UK Biobank | Neuroticism       | 0.069 ± 0.004     |

### Protocol

No Sex / MCI / AD / MMSE numbers reported. Only regression on UK Biobank held-out subjects.

\newpage

# 3. SLIM-Brain

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Not published — preprint, under OpenReview review |
| **arXiv**    | [2512.21881](https://arxiv.org/abs/2512.21881) |

### Architecture

**4D Hiera-JEPA** — Hiera (hierarchical pyramidal ViT, multi-stage) combined with JEPA paradigm. Input = **4D volumes `(T, X, Y, Z)`**. 4D tubelets, global attention within each stage.

### Pretraining datasets

- **HCP** (1 session per subject)
- **CHCP** (Chinese HCP, Ge 2023)
- **AOMIC PIOP1**
- **AOMIC PIOP2**
- **ABCD**
- **Total** — 4,129 sessions (70% of combined data)
- Harmonized to 2mm isotropic / TR 0.72s

### Downstream evaluation datasets

- HCP (Sex, Fingerprint) — internal
- ADNI (MCI classification)
- ADHD-200
- PPMI
- ABIDE (Age classification + regression)

### Results

| Dataset    | Task                  | Acc       | F1 / other    |
|------------|-----------------------|:---------:|:-------------:|
| HCP        | Sex                   | **91.1**  | F1 **91.1**   |
| HCP        | Fingerprint           | 98.5      | F1 98.1       |
| ADNI       | MCI classification    | **69.12** | F1 68.96      |
| ADHD-200   | ADHD                  | 63.53     | —             |
| PPMI       | Parkinson             | 70.40     | —             |
| ABIDE      | Age (classification)  | 64.41     | —             |
| ABIDE      | Age (regression)      | —         | MSE 0.2175    |

### Protocol

70/10/20 train/val/test split, 3 independent runs. Subject-awareness not explicitly stated.

\newpage

# 4. BrainGFM

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Not published — preprint                 |
| **arXiv**    | [2506.02044](https://arxiv.org/abs/2506.02044) |

### Architecture

**Graph Foundation Model** on connectomes. Combines **Graph Contrastive Learning + Graph Masked Autoencoder** + meta-learning + graph prompts + language prompts. Multi-atlas (8 parcellations).

### Pretraining datasets

- **27 datasets** covering 25 disorders
- **25,000+ subjects / 60,000 scans / 400,000 graph samples**
- Identified examples: OpenNeuro, UK Biobank, HCP, ABIDE, ABIDE II, ADHD-200, OASIS, ADNI 2, HBN, SubMex_CUD, UCLA_CNP
- Full list in Appendix N of the paper

### Downstream evaluation datasets

- ADHD-200 (ADHD)
- ABIDE II (Autism)
- **ADNI 2 (Alzheimer)**
- HBN (Depression, Anxiety, OCD, PTSD)
- SubMex_CUD (Cocaine Use Disorder)
- UCLA_CNP (Schizophrenia, Bipolar)

### Results (AUC / Accuracy, %)

| Dataset      | Task                | AUC      | Acc      |
|--------------|---------------------|:--------:|:--------:|
| ADHD-200     | ADHD                | 70.6     | 72.2     |
| ABIDE II     | Autism              | 71.2     | 73.5     |
| **ADNI 2**   | **Alzheimer (AD)**  | **80.3** | **85.1** |
| HBN          | Major Depression    | 83.6     | 85.5     |
| HBN          | Anxiety             | 85.2     | 86.3     |
| HBN          | OCD                 | 80.4     | 85.8     |
| HBN          | PTSD                | 83.2     | 86.3     |
| SubMex_CUD   | Cocaine             | 71.1     | 74.6     |
| UCLA_CNP     | Schizophrenia       | 84.2     | 86.7     |
| UCLA_CNP     | Bipolar             | 73.5     | 76.3     |

### Protocol

Schaefer-100 atlas used in main results. No Sex / Age / Parkinson reported.

\newpage

# 5. LCM (Large Connectome Model)

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — AAAI 2026                    |
| **arXiv**    | [2510.18910](https://arxiv.org/abs/2510.18910) |

### Architecture

**Decoder-only Transformer** (GPT-style) on connectomes (FC matrix). MHSA + MHCA between connectome features and "Brain-Environment-Interaction" tokens (encoding covariates: sex, age, scanner, site).

### Pretraining datasets

| Dataset    | Subjects | Scans  |
|------------|:--------:|:------:|
| HCP-Aging  | 713      | 4,863  |
| HCP-YA     | 248      | 3,293  |
| ADNI       | 138      | 138    |
| PPMI       | 209      | 209    |
| ABIDE      | 1,025    | 1,025  |
| Taowu      | 40       | 40     |
| Neurocon   | 41       | 41     |
| **Total**  | —        | **~10,036** |

### Downstream evaluation datasets

- HCP-Aging (Sex)
- HCP-YA (Sex)
- ABIDE (Sex, Autism)
- ADNI (Alzheimer)
- PPMI (Parkinson)

### Results (F1, %)

| Dataset    | Task             | F1                |
|------------|------------------|:-----------------:|
| HCP-Aging  | Sex              | **73.94 ± 2.45**  |
| HCP-YA     | Sex              | **72.23 ± 1.92**  |
| ABIDE      | Sex              | 87.34 ± 4.48      |
| **ADNI**   | **Alzheimer**    | **85.33 ± 7.35**  |
| PPMI       | Parkinson        | 84.18 ± 11.63     |
| ABIDE      | Autism           | 72.50 ± 1.91      |

### Protocol

Subject-aware 5-fold CV for HCPA/HCPYA/ADNI, 10-fold for others. Pretraining and finetuning always from the same CV fold's training set to prevent leakage.

\newpage

# 6. BNT (Brain Network Transformer)

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — NeurIPS 2022                 |
| **arXiv**    | [2210.06681](https://arxiv.org/abs/2210.06681) |

### Architecture

Pure Transformer on connectome graphs (not a GNN). Each ROI = one token, node features = full connection profile. Key innovation: **Orthonormal Clustering Readout** — pooling via K learned orthogonal prototypes.

### Pretraining

None — supervised end-to-end training.

### Downstream evaluation datasets

- ABIDE (Autism)
- ABCD (Sex, n=7,901)

### Results

| Dataset    | Task                | Metric    | Score        |
|------------|---------------------|-----------|:------------:|
| ABIDE      | Autism              | AUROC     | **80.2%**    |
| ABCD       | Sex (n=7,901)       | —         | not extracted |
| ADNI       | NC vs MCI           | Acc       | **78.90**    |

### Protocol

Subject-level splits. ADNI NC/MCI Acc reported as comparator within Brain-JEPA paper (Brain-JEPA Acc 76.84%, so BNT beats Brain-JEPA on this task).

\newpage

# 7. OViTAD

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — Brain Sciences 2023 (MDPI journal, not a top venue) |
| **arXiv**    | [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full) |

### Architecture

Standard **ViT-B/16** trained on individual **2D fMRI slices** (no 3D, no temporal modeling). Subject-level inference via **majority vote** across slices. No architectural innovation — only hyperparameter tuning.

### Pretraining

None — supervised end-to-end training.

### Downstream evaluation dataset

- ADNI rs-fMRI — 284 subjects total
- Split: 226 train / 27 val / **31 test** (80/10/10 at participant level)

### Results (F1)

| Dataset       | Task        | F1               |
|---------------|-------------|:----------------:|
| ADNI (n=284)  | AD vs HC    | **0.99 ± 0.02**  |
| ADNI (n=284)  | HC vs MCI   | **0.97 ± 0.03**  |

### Protocol

Test set = only 31 subjects. σ=0.02 on n=31 is noise-dominated — interpret with caution.

\newpage

# 8. BrainNetCNN

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — NeuroImage 2017              |
| **arXiv**    | [Kawahara 2017 PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf) |

### Architecture

CNN with **3 custom topology-aware filters** for connectivity matrices:

- **Edge-to-Edge (E2E)** — aggregates row + column per cell
- **Edge-to-Node (E2N)** — aggregates per ROI
- **Node-to-Graph (N2G)** — global pooling

### Pretraining

None — supervised end-to-end training.

### Downstream evaluation dataset (original paper)

- Preterm infant DTI (diffusion MRI, **not fMRI**)
- Task: Bayley-III developmental scores

### Results

| Dataset                  | Task              | Score              |
|--------------------------|-------------------|:------------------:|
| Preterm DTI (orig paper) | Bayley-III scores | **not comparable** |

### Protocol

Original architecture is now widely reused as a supervised baseline on adult fMRI by BNT / BrainGFM / BrainGB.

\newpage

# 9. SwiFT

|              |                                          |
|--------------|------------------------------------------|
| **Status**   | Published — NeurIPS 2023                 |
| **arXiv**    | [2307.05916](https://arxiv.org/abs/2307.05916) |

### Architecture

**4D Swin Transformer** on raw fMRI volumes (no atlas). 4D windows `(X, Y, Z, T)` + windowed self-attention + shifted windows between layers + hierarchical (4 stages). Closest volumetric Transformer paradigm to ours.

### Pretraining

Light contrastive SSL (composition not specified in main paper).

### Downstream evaluation datasets

- HCP-YA (Sex, Age, Cognitive intelligence)
- ABCD (Sex, Age, Intelligence)
- UK Biobank (Sex, Age)

### Results (extracted from comparator tables in Brain-JEPA & SLIM-Brain)

| Dataset    | Task              | Acc       | Other        |
|------------|-------------------|:---------:|:------------:|
| OASIS-3    | AD Conversion     | 65.00     | F1 66.80     |
| ADNI       | MCI               | 64.45     | —            |
| ABIDE      | Age (cls)         | 62.22     | MSE 0.4137   |
| ADHD-200   | ADHD              | 60.81     | —            |
| PPMI       | Parkinson         | 58.10     | —            |
| HCP / ABCD / UKB | Sex, Age    | —         | not extracted |

### Protocol

Numbers reported as comparator within Brain-JEPA Table 9 and SLIM-Brain Table 1. Original SwiFT-paper numbers on HCP/ABCD/UKB not extracted in this pass.

\newpage

# Datasets — dimensions and access

| Dataset       | # subj.    | TR (s) | Typical T | Access mode                              |
|---------------|:----------:|:------:|:---------:|------------------------------------------|
| UK Biobank    | 40,000+    | 0.735  | 490       | Paid DUA (~£500-1000), 2-6 months        |
| HCP-YA        | 1,200      | 0.72   | 1,200     | Open with DUA, fast                      |
| HCP-Aging     | 700        | 0.8    | ~470      | NDA controlled, 1-3 months               |
| CHCP          | ~370       | 0.72   | varies    | Chinese open access                      |
| AOMIC PIOP1   | 216        | 2.0    | ~480      | Open                                     |
| AOMIC PIOP2   | 226        | 2.0    | ~480      | Open                                     |
| ABCD          | 11,800+    | 0.8    | ~375      | NDA controlled, 1-3 months               |
| ADNI / ADNI 2 | ~1,000     | 3.0    | ~140      | DUA, 1-2 weeks                           |
| ABIDE I       | 1,112      | 1.5-3  | 76-300    | Open (NITRC / PCP)                       |
| ABIDE II      | 1,114      | varies | varies    | Open                                     |
| ADHD-200      | 776        | 1.5-2.5| 70-280    | Open (NITRC)                             |
| PPMI          | ~500       | 2.4    | ~200      | DUA, 1-2 weeks                           |
| HBN           | 3,000+     | 0.8    | varies    | Open (AWS S3)                            |
| OASIS-3       | 1,098      | 2.2    | ~180      | Registration + proposal, 1-7 days        |
| MACC          | restricted | varies | varies    | Restricted (Asian collaborators)         |
| CamCAN        | 700        | 1.97   | 261       | Open with registration                   |
| SubMex_CUD    | restricted | varies | varies    | Restricted                               |
| UCLA_CNP      | 272        | 2.0    | 152       | Open (OpenNeuro)                         |
| Taowu         | 40         | 2.0    | ~120      | Open (OpenNeuro)                         |
| Neurocon      | 41         | 2.0    | ~140      | Open (OpenNeuro)                         |

### Access categories (in order of effort)

- **Open access** (immediate via Python libs like `nilearn`): ABIDE I/II, ADHD-200, AOMIC PIOP1/PIOP2, CHCP, HBN, UCLA_CNP, Taowu, Neurocon
- **Registration + short proposal** (1-7 days): OASIS-3, CamCAN
- **DUA, fast** (1-2 weeks): ADNI, PPMI, HCP-YA
- **NDA controlled** (1-3 months): HCP-Aging, ABCD
- **Paid DUA, slow** (2-6 months): UK Biobank (~£500-1000)
- **Restricted to specific collaborators**: MACC, SubMex_CUD

\newpage

# Summary tables

### Pretraining datasets vs evaluation datasets

| Model        | Pretraining datasets                                  | Evaluation datasets                                 |
|--------------|--------------------------------------------------------|------------------------------------------------------|
| Brain-JEPA   | UK Biobank                                             | UKB, HCP-Aging, ADNI, MACC, OASIS-3, CamCAN          |
| BrainLM      | UK Biobank + HCP                                       | UK Biobank only                                      |
| SLIM-Brain   | HCP + CHCP + AOMIC PIOP1/PIOP2 + ABCD                  | HCP, ADNI, ADHD-200, PPMI, ABIDE                     |
| BrainGFM     | 27 datasets (HCP, UKB, ABIDE, ADHD, OASIS, ADNI 2, HBN, ...) | ADNI 2, ABIDE II, ADHD-200, HBN, SubMex_CUD, UCLA_CNP |
| LCM          | HCPA + HCPYA + ADNI + PPMI + ABIDE + Taowu + Neurocon  | HCPA, HCPYA, ABIDE, ADNI, PPMI                       |
| BNT          | None (supervised end-to-end)                           | ABIDE, ABCD                                          |
| OViTAD       | None (supervised end-to-end)                           | ADNI (284 subjects)                                  |
| BrainNetCNN  | None (supervised end-to-end)                           | Preterm DTI (original paper)                         |
| SwiFT        | Light contrastive SSL (corpus unspecified)             | HCP, ABCD, UK Biobank                                |

### Labels used per model

| Model        | Labels reported                                                                            |
|--------------|---------------------------------------------------------------------------------------------|
| Brain-JEPA   | DX (NL/MCI), Amyloid β +/-, Sex, Depression                                                 |
| BrainLM      | Age, PTSD, Anxiety, Neuroticism (z-scored MSE regression)                                   |
| SLIM-Brain   | DX (MCI), Sex, Fingerprint, ADHD, Parkinson, Age                                            |
| BrainGFM     | ADHD, ASD, AD (DX), MDD, Anxiety, OCD, PTSD, Cocaine, Schizo, Bipolar (10 disorders)        |
| LCM          | Sex, DX (AD, PD, ASD, SZ)                                                                   |
| BNT          | DX (Autism), Sex                                                                             |
| OViTAD       | DX (AD vs HC, HC vs MCI)                                                                     |
| BrainNetCNN  | Bayley-III scores (preterm)                                                                 |
| SwiFT        | Sex, Age, Cognitive intelligence                                                             |

\newpage

# Notes on comparability

- **Most SOTA papers report Accuracy (%) and F1 (%)**, not AUC. Direct AUC-to-Acc conversion is approximate.
- **Brain-JEPA / BrainGFM / LCM use DX clinical diagnosis labels** (NL / MCI / AD from ADNI metadata). Direct comparison to CDR-based binarisations requires label alignment.
- **No SOTA paper reports horizon-specific cognitive decline forecasting** (1Y / 2Y / 3Y splits). Closest comparator: Brain-JEPA OASIS-3 AD Conversion (single horizon, Acc 69.00 / F1 67.32).
- **Subject-level splits**: Brain-JEPA (6:2:2), SLIM-Brain (70:10:20), LCM (subject-aware 5/10-fold), OViTAD (80:10:10 at participant level). Subject-awareness explicitly confirmed only for LCM and OViTAD.

# Sources

1. **Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking** — Dong & Li et al., NeurIPS 2024 Spotlight — [arxiv.org/abs/2409.19407](https://arxiv.org/abs/2409.19407)
2. **BrainLM: A foundation model for brain activity recordings** — Ortega Caro et al., ICLR 2024 — [biorxiv 10.1101/2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)
3. **SLIM-Brain: A Data- and Training-Efficient Foundation Model for fMRI Data Analysis** — Anonymous, arXiv Dec 2025 — [arxiv.org/abs/2512.21881](https://arxiv.org/abs/2512.21881)
4. **Brain Graph Foundation Model (BrainGFM)** — arXiv 2025 — [arxiv.org/abs/2506.02044](https://arxiv.org/abs/2506.02044)
5. **Large Connectome Model (LCM)** — AAAI 2026 — [arxiv.org/abs/2510.18910](https://arxiv.org/abs/2510.18910)
6. **Brain Network Transformer (BNT)** — Kan, Cui et al., NeurIPS 2022 — [arxiv.org/abs/2210.06681](https://arxiv.org/abs/2210.06681)
7. **OViTAD** — Sarraf et al., Brain Sciences 2023 — [biorxiv 10.1101/2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
8. **BrainNetCNN** — Kawahara et al., NeuroImage 2017 — [PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
9. **SwiFT** — Kim et al., NeurIPS 2023 — [arxiv.org/abs/2307.05916](https://arxiv.org/abs/2307.05916)
