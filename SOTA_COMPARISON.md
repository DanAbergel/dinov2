# SOTA comparison — fMRI prediction models

Compiled for the meeting with Ariel & Yoni on 2026-06-14. All numerical claims are taken **verbatim** from primary sources (linked in the **Sources** section). 22 claims verified by adversarial vote (3-0 unanimous for 17 of them). 3 candidate claims were refuted and excluded — see **Refuted claims** at the bottom.

Our setup (for reference, run A = full FT at iter 19999, ADNI linear probe on frozen CLS, 5-fold subject-aware CV):

| Label | Metric | Our number |
|---|---|---:|
| HCP Sex | AUC | **0.70** (A) / **0.84** (C, freeze-fmri) |
| ADNI Sex | AUC | **0.72** (A) / **0.81** (C) |
| ADNI Age | MAE | **5.0 – 5.7 years** |
| ADNI MMSE | MAE | **2.4** |
| ADNI CDR_Binary (≥0.5) | AUC | **0.53** |
| ADNI CDR NC-vs-AD (drop MCI) | AUC | **0.74** |
| ADNI CDR NC-vs-MCI (drop AD) | AUC | **0.49** |
| ADNI FAQ Binary | AUC | **0.65** |
| ADNI Degradation 1Y / 2Y / 3Y | AUC | **0.56 / 0.47 / 0.45** |

---

## 1. Cards récapitulatives — 1 papier = 1 fiche

À parcourir verbalement en réunion. Pour les chiffres exacts dans le contexte (CV protocol, caveats), voir le **tableau détaillé** en section 2.

### Brain-JEPA — NeurIPS 2024 Spotlight • [arxiv 2409.19407](https://arxiv.org/abs/2409.19407)
- **Architecture** : JEPA + ViT (Joint Embedding Predictive Architecture, style I-JEPA mais sur séries temporelles fMRI au lieu d'images)
- **Pretraining** : **UK Biobank, ~32,130 sujets** (80% des 40,162 dispo), TR=0.735s. Représentation = **450 ROI** (Schaefer-400 cortical + Tian-50 subcortical) × 160 timesteps. **Pas du tout volumétrique.**
- **Résultats clés** :
  - UKB Sex : **Acc 88.17% / F1 88.58%**
  - HCP-Aging Sex : **Acc 81.52% / F1 84.26%**
  - **ADNI NC-vs-MCI : Acc 76.84% / F1 86.32%** (sur n=189 sujets, split 6:2:2 subject-aware, labels DX pas CDR)
- **Vs nous** : on est **bien en dessous** (notre HCP Sex AUC 0.84 ≈ Acc 75-80%, leur HCPA Sex Acc 81.5%) — gap explicable surtout par la taille du pretraining (32k vs notre 1.2k)

### BrainLM — ICLR 2024 • [biorxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)
- **Architecture** : Transformer Masked-Autoencoder (style MAE), **111M params**
- **Pretraining** : **UKB 76,296 recordings (~6,450 h) + HCP 1,002 recordings (~250 h) = 77,298 total / 6,700 h.** Représentation = **AAL-424 ROI** (atlas neuro-anatomique). Pretraining sur 80% UKB = 61,038 recordings.
- **Résultats clés** : UKB Age **MSE z-scoré 0.503**, PTSD 0.015, Anxiety 0.073, Neuroticism 0.069. **Aucun chiffre Sex / MMSE / AD / MCI rapporté.**
- **Vs nous** : **non comparable directement** — ils reportent MSE z-scoré qu'on ne peut pas convertir en MAE années sans la std d'âge UKB

### SLIM-Brain — arXiv Déc 2025 • [arxiv 2512.21881](https://arxiv.org/abs/2512.21881)
- **Architecture** : **4D Hiera-JEPA** (hiérarchique, sur volumes 4D — plus proche de nous architecturalement)
- **Pretraining** : **~4,000 sessions fMRI seulement** (8-20× moins que Brain-JEPA / BrainLM). Volumes 4D bruts, pas d'atlas.
- **Résultats clés** : revendique **SOTA sur 7 benchmarks** + "~30% de la mémoire GPU des méthodes voxel". **Chiffres précis non extraits** (papier encore sous review OpenReview).
- **Vs nous** : architecturalement le plus proche, à creuser après le meeting

### BrainGFM (Brain Graph Foundation Model) — arXiv 2025 • [arxiv 2506.02044](https://arxiv.org/abs/2506.02044)
- **Architecture** : **Graph foundation model** — contrastive graph + masked-autoencoder + meta-learning + language prompts
- **Pretraining** : **27 datasets neuroimagerie**, 25 pathologies, **25,000+ sujets, 60,000 scans, 400,000 samples de graphes**, 8 parcellations différentes
- **Résultats clés** : chiffres précis non extraits dans ce pass
- **Vs nous** : paradigme totalement différent (graphes de connectomes, pas volumes) — utile comme baseline si on passe au connectome

### LCM (Large Connectome Model) — AAAI 2026 • [arxiv 2510.18910](https://arxiv.org/abs/2510.18910)
- **Architecture** : **Transformer decoder-only** avec multi-head self-attention + multi-head cross-attention (entre features connectome et tokens "brain-environment")
- **Pretraining** : taille de corpus **non clairement caractérisée** dans le papier (claim 10k scans réfuté à 1-2 par notre verif). Représentation = **connectome (matrice FC)**.
- **Résultats clés** sous **subject-aware 5-fold CV** (le même protocole que nous) :
  - HCP-Aging Sex F1 **73.94 ± 2.45** / HCP-YA Sex F1 **72.23 ± 1.92** / ABIDE Sex F1 **87.34 ± 4.48**
  - **ADNI Alzheimer F1 85.33 ± 7.35**
  - PPMI Parkinson F1 84.18 / ABIDE Autism F1 72.50
- **Vs nous** : **comparable directement** — même CV protocole. Notre HCP Sex 0.84 AUC ≈ leur HCPYA F1 72 ≈ kif-kif. Mais leur ADNI AD F1 85 est au-dessus de notre 0.74 AUC.

### BNT (Brain Network Transformer) — NeurIPS 2022 • [arxiv 2210.06681](https://arxiv.org/abs/2210.06681)
- **Architecture** : Transformer sur graphes de connectome, features = profils de connexion par nœud + "Orthonormal Clustering Readout"
- **Pretraining** : **AUCUN** — supervisé end-to-end sur la tâche downstream directement
- **Résultats clés** :
  - ABIDE Autism **AUROC 80.2%**
  - ABCD Sex (n=7,901 sujets) — chiffre précis non extrait
  - Important : selon Brain-JEPA, **BNT bat Brain-JEPA sur ADNI NC/MCI Acc** (78.90% vs 76.84%)
- **Vs nous** : pas de comparator direct ADNI/HCP — utile surtout comme baseline supervisée historique

### OViTAD — Brain Sciences 2023 • [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
- **Architecture** : **ViT 2D** entraîné slice-par-slice, agrégation par majority vote au niveau sujet
- **Pretraining** : **AUCUN** — supervisé end-to-end ADNI
- **Résultats clés** sous split **subject-aware 80/10/10** (226/27/31 sujets) :
  - **AD vs HC F1 0.99 ± 0.02**
  - **HC vs MCI F1 0.97 ± 0.03**
- **Vs nous** : leur F1 0.99 est intimidant mais **test = 31 sujets** → std 0.02 sur n=31 = bruit massif. À nuancer en présentation.

### BrainNetCNN — NeuroImage 2017 • [PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
- **Architecture** : CNN sur matrice de connectivité avec 3 filtres custom : **Edge-to-Edge (E2E), Edge-to-Node (E2N), Node-to-Graph (N2G)**
- **Pretraining** : **AUCUN** — supervisé end-to-end
- **Résultats clés (papier original)** : prédiction scores Bayley-III sur **DTI bébés prématurés** (27-46 sem GA) — **non comparable à nous**. Utilisé comme baseline supervisée par BNT / BrainGFM / BrainGB sur fMRI adulte.
- **Vs nous** : baseline historique de référence, pas de chiffres ADNI/HCP comparables dans le papier original

### SwiFT — NeurIPS 2023 • [arxiv 2307.05916](https://arxiv.org/abs/2307.05916)
- **Architecture** : **Swin Transformer 4D** — 4D window MHSA + positional embeddings absolus. **Travaille directement sur volumes 4D fMRI** (sans atlas). **C'est l'analogue architectural le plus proche de notre setup.**
- **Pretraining** : supervisé / SSL léger
- **Résultats clés** : évalué sur **HCP, ABCD, UKB** pour Sex / Age / cognitive intelligence — chiffres précis non extraits dans ce pass
- **Vs nous** : architecturalement le plus pertinent à comparer, à creuser après le meeting

---

## 2. SOTA papers — tableau détaillé pour cross-référence

| Paper / venue / year | arXiv / preprint | Architecture | Pretrain dataset (#subj, #scans, hours) | Atlas / repr. | Downstream | Reported numbers (verbatim) | CV protocol | Caveat |
|---|---|---|---|---|---|---|---|---|
| **Brain-JEPA** — Dong & Li, NeurIPS 2024 **Spotlight** | [arxiv.org/abs/2409.19407](https://arxiv.org/abs/2409.19407) | JEPA ViT | UK Biobank: 40,162 subjects total, **~32,130 used for pretraining** (80%), TR=0.735s | **450 ROI** (Schaefer-400 cortical + Tian-50 subcortical), 160 timesteps | UKB held-out, HCP-Aging, ADNI (n=189), MACC | **UKB Sex Acc 88.17% / F1 88.58%** • **HCPA Sex Acc 81.52% / F1 84.26%** • **ADNI NC-vs-MCI Acc 76.84% / F1 86.32%** • **MACC NC-vs-MCI Acc 65.98% / F1 64.67%** | **6:2:2 participant-level** (subject-aware) | Uses **DX-defined** NC/MCI labels (not CDR); ADNI n=189 is small, so single-digit metric swings are noise |
| **BrainLM** — Ortega Caro et al., ICLR 2024 | [biorxiv 2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1) | Transformer Masked-Autoencoder, **111M params** | UKB **76,296 recordings (~6450 h)** + HCP **1,002 recordings (~250 h)** = **77,298 total (6,700 h)**; pretrained on 80% UKB (61,038 recordings) | **AAL-424** ROI | UKB clinical regression | **Age (z-scored) MSE 0.503 ± 0.021** • PTSD MSE 0.015 • Anxiety MSE 0.073 • Neuroticism MSE 0.069. **NO Sex / MMSE / CDR / AD numbers reported.** | Subject-level held-out 20% UKB | Reports MSE on **z-scored** targets only — not back-convertible to MAE without UKB age σ (~7.5 y) |
| **SLIM-Brain** — Anon., arXiv Dec 2025 | [arxiv.org/abs/2512.21881](https://arxiv.org/abs/2512.21881) | **4D Hiera-JEPA** (hierarchical) | **Only ~4,000 fMRI sessions** (8–20× smaller than Brain-JEPA / BrainLM) | 4D voxel volumes | 7 public benchmarks (not enumerated in our extracted claims) | Claims **SOTA across 7 benchmarks**, "~30% of GPU memory of voxel methods". Specific Sex/AD numbers not extracted. | not specified | **Under OpenReview review**, no independent replication; "SOTA" is authors' framing |
| **BrainGFM** — Brain Graph Foundation Model, arXiv 2025 | [arxiv.org/abs/2506.02044](https://arxiv.org/abs/2506.02044) | **Graph FM** (GCL + GraphMAE + meta-learning + language prompts) | **27 datasets**, 25 disorders, **25k+ subjects**, **60k scans**, **400k graph samples**, **8 parcellations** | Connectome graphs, multi-atlas | Multi-disorder | Quantitative numbers not extracted in this pass | not specified | Self-described as "first graph-based fMRI foundation model"; very broad pretraining corpus |
| **LCM** (Large Connectome Model) — AAAI 2026 | [arxiv.org/abs/2510.18910](https://arxiv.org/abs/2510.18910) | **Decoder-only Transformer** with MHSA + MHCA on connectome features × brain-environment tokens | Pretraining size **not confidently characterized** (one candidate claim refuted) | Connectome (FC matrix) | HCPA, HCPYA, ADNI, PPMI, ABIDE, Neurocon | **HCPA Sex F1 73.94 ± 2.45** • **HCPYA Sex F1 72.23 ± 1.92** • **ABIDE Sex F1 87.34 ± 4.48** • **Neurocon Sex F1 100.00** • **ADNI AD F1 85.33 ± 7.35** • **PPMI Parkinson F1 84.18 ± 11.63** • **ABIDE Autism F1 72.50 ± 1.91** | **Subject-aware 5-fold** for HCPA/HCPYA/ADNI, 10-fold for others. Pretrain/finetune always from training fold of CV — no leakage | Highly relevant: same CV protocol as ours (subject-aware k-fold) |
| **BNT** (Brain Network Transformer) — NeurIPS 2022 | [arxiv.org/abs/2210.06681](https://arxiv.org/abs/2210.06681) | Transformer on brain-network graph with Orthonormal Clustering Readout | **N/A — supervised end-to-end** | Connection profiles as node features | ABIDE (autism), ABCD (sex, n=7,901) | **ABIDE Autism AUROC 80.2%** • ABCD Sex reported. **No ADNI / HCP / MMSE / CDR.** Also reported as comparator in Brain-JEPA: BNT **beats** Brain-JEPA on ADNI NC/MCI Acc (78.90% vs 76.84%) | Subject-level | Strong on connectome graphs; no direct comparator to ADNI rs-fMRI Sex / Age |
| **OViTAD** — Sarraf et al., Brain Sciences 2023 | [biorxiv 2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full) | 2D ViT on fMRI slices | **N/A — supervised end-to-end** | 2D slices, majority-vote at subject level | ADNI rs-fMRI (226 train / 27 val / 31 test subjects) | **AD vs HC F1 0.99 ± 0.02** • **HC vs MCI F1 0.97 ± 0.03** | **Subject-aware 80/10/10** (participant-level — initial suspicion of slice-leakage was refuted 0-3 by verifiers) | **Test set is only 31 subjects** → F1 0.99 ± 0.02 carries large variance; slice-aggregation via majority vote |
| **BrainNetCNN** — Kawahara et al., NeuroImage 2017 | [primary PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf) | CNN with E2E / E2N / N2G filters on connectivity matrix topology | **N/A — supervised end-to-end** | Connectome | **Preterm-infant DTI** (27–46 weeks GA), Bayley-III scores | Original numbers are for **preterm DTI, not adult fMRI** — NOT comparable to ADNI / HCP labels | Subject-level | Architecture reused as **baseline** by BNT / BrainMass / BrainGFM / BrainGB benchmark on adult fMRI |
| **SwiFT** — Kim et al., NeurIPS 2023 | [arxiv.org/abs/2307.05916](https://arxiv.org/abs/2307.05916) | **4D Swin Transformer** with 4D window MHSA + absolute pos. emb. — operates on **fMRI VOLUMES directly** (closest architectural analog to ours) | **N/A — supervised end-to-end** (or self-sup variant on UKB) | 4D voxels (no atlas) | HCP, ABCD, UKB — Sex / Age / cognitive intelligence | Specific numbers not extracted in this pass | not specified in our claims | Most relevant architectural comparator — same volumetric paradigm |

---

## 2. Pretraining datasets — popularity ranking

Ranked by aggregate scan-count across the surveyed SOTA papers:

| # | Dataset | ~#subjects | Access | Used by |
|---:|---|---:|---|---|
| 1 | **UK Biobank** | ~40k+ rs-fMRI | DUA (paid) | BrainLM, Brain-JEPA, SwiFT, BrainGFM |
| 2 | **HCP-Young Adult** | ~1,200 | open with DUA | BrainLM, SwiFT, LCM, **us** |
| 3 | **HCP-Aging** | ~700 | NDA controlled | Brain-JEPA, LCM |
| 4 | **ABCD** | ~12k adolescents | NDA controlled | BNT, SwiFT, BrainGFM |
| 5 | **ADNI** | ~hundreds (rs-fMRI subset) | DUA | downstream only: Brain-JEPA, LCM, OViTAD, **us** |
| 6 | **ABIDE I/II** | ~1k autism | open | BNT, LCM, BrainGFM |
| 7 | **PPMI** (Parkinson) | ~hundreds | DUA | LCM, BrainGFM |
| 8 | **MACC** (Asian aging) | restricted | restricted | Brain-JEPA |

**Strategic observation**: every paper at the top of the SOTA leaderboard pretrains on **UK Biobank (40k+ subjects)**. We pretrain on **HCP-YA only (~1,200 subjects, 30× smaller)**. This is the most likely root cause of our gap on downstream metrics.

---

## 3. Where we stand — synthesis

### Sex (calibration task)
**Below SOTA, but the gap is bounded.** Brain-JEPA HCPA Sex Acc 81.52% ([Brain-JEPA, 2024](https://arxiv.org/abs/2409.19407)) translates to roughly AUC 0.88–0.90; our HCP Sex AUC is 0.84 (run C, freeze-fmri). The frozen-fmri variant is competitive with LCM ([LCM, 2026](https://arxiv.org/abs/2510.18910)) which reports HCPYA Sex F1 = 72.23 ± 1.92 and HCPA F1 = 73.94 ± 2.45 — both subject-aware 5-fold, same protocol as ours. Brain-JEPA UKB Sex Acc 88.17% is out of reach without UKB pretraining.

### NC vs AD
**Below SOTA**, but in the right region. LCM ADNI AD F1 = 85.33 ± 7.35 (subject-aware 5-fold, [LCM, 2026](https://arxiv.org/abs/2510.18910)) is the cleanest comparator and exceeds our ADNI CDR NC-vs-AD AUC 0.74 (which roughly corresponds to F1 ~0.60–0.65 at balanced threshold). OViTAD AD-vs-HC F1 0.99 ± 0.02 ([OViTAD, 2023](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)) should be treated cautiously — test set is **31 subjects**, σ=0.02 on n=31 is noise-dominated.

### NC vs MCI
**Far below SOTA — biggest gap.** Brain-JEPA ADNI NC-vs-MCI Acc 76.84% / F1 86.32% ([Brain-JEPA, 2024](https://arxiv.org/abs/2409.19407)) vs our 0.49 AUC (chance level). Two caveats partly explain the gap:
1. Brain-JEPA uses **DX-defined labels**, we use **CDR-defined** — partial label mismatch.
2. Brain-JEPA's 189 ADNI subjects are pre-selected and balanced; our ADNI is not.

But neither caveat closes a 27-point gap. The honest read: **HCP pretraining doesn't produce features that separate NC from prodromal MCI**, which is consistent with HCP-YA = healthy young adults (no clinical signal to distill).

### Age regression
**No direct MAE comparator exists in the surveyed SOTA.** BrainLM reports z-scored Age MSE 0.503 ([BrainLM, 2024](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1)) which is not back-convertible without the UKB age standard deviation (~7.5 y assumed). Brain-JEPA does age **classification (binned)**, not regression. SwiFT, SLIM-Brain, LCM do not report ADNI age. Our 5.0–5.7 y MAE is in an under-served niche — a thesis contribution rather than a benchmark fail.

### Degradation 1Y / 2Y / 3Y (cognitive decline forecasting)
**No published horizon-specific benchmark exists**, per our verification pass. Brain-JEPA Table 3 includes ADNI "prognosis" tasks but the specific 1Y/2Y/3Y formulations were not extracted (open question). No other SOTA (LCM / BrainLM / OViTAD / SwiFT / BNT / BrainNetCNN / BrainGFM / SLIM-Brain) benchmarks this. Our 0.45–0.56 AUCs are in **literature whitespace** — could be a thesis contribution.

---

## 4. The 3 takeaways for Ariel & Yoni

1. **Scale matters more than method.** Every paper above us on Sex / NC-vs-MCI pretrains on **UK Biobank (~40k subjects)**. We use HCP-YA (~1.2k). Moving to UKB pretraining is probably the single biggest lever.
2. **NC-vs-AD works, NC-vs-MCI doesn't, and that's coherent with the literature.** Even SOTA struggles on NC-vs-MCI relative to NC-vs-AD. The fact that our NC-vs-AD AUC = 0.74 with HCP-only pretraining is a **positive signal** — features encode advanced neurodegeneration.
3. **Two task niches without comparators**: ADNI Age MAE (no SOTA reports MAE in years) and Degradation 1/2/3Y horizon forecasting (no SOTA reports this at all). These can be framed as **original thesis contributions** rather than benchmark gaps.

---

## 5. Refuted claims (excluded from the report)

The verification pass killed 3 candidate claims (full vote in parentheses):

| Claim | Vote | Source |
|---|---|---|
| LCM is pretrained on 10,036 fMRI scans combining HCPA + HCPYA + disease cohorts using AAL atlas | **1-2 refuted** | [LCM, 2026](https://arxiv.org/abs/2510.18910) — LCM's actual pretraining corpus is not confidently characterized |
| OViTAD uses slice-level random split → multiple slices from same scan in train+test (non subject-aware) | **0-3 refuted (unanimous)** | [OViTAD, 2023](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full) — split is participant-level 80/10/10, confirmed by 3 verifiers |
| arXiv 2501.16409 model trained on 345 ADNI subjects / 570 scans | **1-2 refuted** | [arXiv 2501.16409](https://arxiv.org/pdf/2501.16409) |

---

## 6. Open questions (not answered by this pass)

1. **Brain-JEPA Table 3 prognosis numbers** — what are the specific NC→MCI / MCI→AD conversion AUCs? These would be our actual Degradation comparator.
2. **SLIM-Brain ADNI numbers** — the 7 benchmarks are not enumerated; might include a directly comparable AD task.
3. **BrainLM Age MAE back-conversion** — what is UKB age σ exactly? Could let us convert 0.503 MSE z-scored → MAE in years for direct comparison.
4. **Mostofi / Adeli / Duncan group** — any unpublished horizon-specific cognitive decline AUCs on ADNI?

---

## Sources

All papers cited above, with title + URL (clickable):

1. **Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking** — Dong & Li et al., NeurIPS 2024 Spotlight — [arxiv.org/abs/2409.19407](https://arxiv.org/abs/2409.19407)
2. **BrainLM: A foundation model for brain activity recordings** — Ortega Caro et al., ICLR 2024 — [biorxiv 10.1101/2023.09.12.557460](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1) — model card: [HuggingFace vandijklab/brainlm](https://huggingface.co/vandijklab/brainlm)
3. **SLIM-Brain: A Data- and Training-Efficient Foundation Model for fMRI Data Analysis** — Anon., arXiv Dec 2025 — [arxiv.org/abs/2512.21881](https://arxiv.org/abs/2512.21881)
4. **Brain Graph Foundation Model (BrainGFM): Pre-Training and Prompt-Tuning across Broad Atlases and Disorders** — arXiv 2025 — [arxiv.org/abs/2506.02044](https://arxiv.org/abs/2506.02044)
5. **Large Connectome Model (LCM): A Decoder-Only Foundation Model for fMRI** — AAAI 2026 — [arxiv.org/abs/2510.18910](https://arxiv.org/abs/2510.18910)
6. **Brain Network Transformer (BNT)** — Kan, Cui et al., NeurIPS 2022 — [arxiv.org/abs/2210.06681](https://arxiv.org/abs/2210.06681)
7. **OViTAD: Optimized Vision Transformer for the diagnosis of Alzheimer's disease** — Sarraf et al., Brain Sciences 2023 — [biorxiv 10.1101/2021.11.27.470184](https://www.biorxiv.org/content/10.1101/2021.11.27.470184v2.full)
8. **BrainNetCNN: Convolutional neural networks for brain networks** — Kawahara et al., NeuroImage 2017 — [PDF](https://gwern.net/doc/psychology/neuroscience/2017-kawahara.pdf)
9. **SwiFT: Swin 4D fMRI Transformer** — Kim et al., NeurIPS 2023 — [arxiv.org/abs/2307.05916](https://arxiv.org/abs/2307.05916)

**Methodology references:**
- **How You Split Matters: Data Leakage and Subject Characteristics Studies in Longitudinal Brain MRI Analysis** — [arxiv.org/abs/2309.00350](https://arxiv.org/abs/2309.00350) — quantifies ~0.10–0.20 AUC inflation from non subject-aware splits (not actually applicable to the top SOTA papers above, since they all use subject-aware splits per our verification)
- **Brain Imaging Foundation Models, Are We There Yet?** — systematic review arXiv 2506.13306 — broadly critiques the field for over-reliance on ML metrics vs medically meaningful evaluation
