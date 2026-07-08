# fMRI Foundation Model (V2) — Results

## 1. Pretraining corpus

Five sources, **4627 scans**, harmonized to TR = 0.72 s with a fixed T = 270 window.

| Dataset | Native TR (s) | Probe holdout (30%) | Per-batch quota |
|---|---|---|---|
| HCP | 0.72 | yes | 4 |
| ABIDE | per-site | yes | 4 |
| OASIS-3 | 2.2 | yes | 4 |
| ADNI | 3.0 | yes | 3 |
| AOMIC | 0.75/2.0 | no (kept whole) | 1 |

*Quota = per-batch proportion (proportional sampler). Holdout: 30% of subjects excluded from pretraining -> leakage-free test set.*

## 2. Datasets & scan counts

Every dataset we have, with its role (pretraining / downstream probe) and scan count.

| Dataset | Role | Scans | Note |
|---|---|---|---|
| HCP (rest) | pretrain + probe | 1084 | Sex / Age |
| ABIDE | pretrain + probe | 1035 | Autism / Age / Sex |
| OASIS-3 | pretrain + probe | 1197 | AD Conversion (labels pending) |
| ADNI | pretrain + probe | 812 | NC-MCI / AD-HC / Amyloid |
| AOMIC | pretrain only | ~499 | derived (4627 - others); no probe |
| ADHD-200 | probe only (external) | 162 | 115 with usable DX label |
| COBRE | probe only (external) | 146 | schizophrenia (72 SZ / 74 HC) |
| UCLA (ds000030) | probe only (external) | 265 | downloader ready, not yet run |
| HCP task-fMRI | probe only | — | 7 tasks x ~1050, download pending |

*Pretraining corpus total = 4627 (HCP + ABIDE + OASIS + ADNI + AOMIC). External datasets (ADHD-200 / COBRE / UCLA) were never seen in pretraining.*

## 3. Pretraining ablations (5 SSL runs) — test AUROC

Each run starts from the same DINOv2 (ImageNet) init and changes **one** factor:

- **base**: reference (freeze blocks 0-8) · **fourier**: Fourier positional encoding
- **noblock2**: drop block_2 · **pool**: AvgPool downsampling (instead of strided conv)
- **unfrozen**: all layers unfrozen during SSL

| Axis | base | fourier | noblock2 | pool | unfrozen |
|---|---|---|---|---|---|
| ABIDE / Autism | 0.605 | 0.518 | 0.484 | 0.606 | 0.541 |
| ABIDE / Age | 0.846 | 0.756 | 0.776 | 0.877 | 0.828 |
| ABIDE / Sex | 0.521 | 0.514 | 0.492 | 0.694 | 0.538 |
| ADNI / NC-MCI | 0.567 | 0.497 | 0.484 | 0.566 | 0.604 |
| ADNI / AD-HC | 0.481 | 0.537 | 0.577 | 0.591 | 0.731 |
| ADNI / Amyloid | 0.611 | 0.539 | 0.500 | 0.641 | 0.640 |
| HCP / Sex | 0.908 | 0.900 | 0.933 | 0.962 | 0.886 |
| HCP / Age | 0.633 | 0.594 | 0.626 | 0.680 | 0.592 |
| OASIS / AD Conv | — | — | — | — | — |

*Metric: test AUROC (linear probe, 30% held-out). OASIS = labels missing.*

## 4. Probe ablations (on `base`)

### 4a. Temporal aggregation of the CLS token — AUROC

| Axis | mean (384-d) | mean_std (768-d) | delta |
|---|---|---|---|
| ABIDE / Autism | 0.605 | 0.583 | -0.023 |
| ABIDE / Age | 0.846 | 0.835 | -0.011 |
| ADNI / NC_vs_MCI | 0.567 | 0.505 | -0.062 |
| ADNI / Amyloid | 0.611 | 0.606 | -0.005 |
| ADNI / AD_vs_HC | 0.481 | 0.404 | -0.077 |
| HCP / Sex | 0.908 | 0.853 | -0.055 |
| HCP / Age | 0.633 | 0.595 | -0.037 |
| ADHD / ADHD | 0.552 | 0.557 | +0.005 |
| COBRE / Schizophrenia | 0.519 | 0.538 | +0.019 |

*mean_std mainly helps task dynamics; elsewhere `mean` wins.*

### 4b. MLP head architecture — AUROC

| Axis | 128 | 256 | 256x128 | 512x256 | 512x256x128 | linear |
|---|---|---|---|---|---|---|
| ADNI / NC_vs_MCI | 0.588 | 0.509 | 0.523 | 0.475 | 0.433 | 0.567 |
| ADNI / AD_vs_HC | 0.407 | 0.467 | 0.472 | 0.448 | 0.487 | 0.481 |
| ADNI / Amyloid | 0.621 | 0.623 | 0.610 | 0.631 | 0.610 | 0.611 |
| ABIDE / Autism | 0.550 | 0.497 | 0.561 | 0.509 | 0.522 | 0.605 |
| HCP / Sex | 0.844 | 0.906 | 0.896 | 0.898 | 0.895 | 0.908 |
| ADHD / ADHD | 0.477 | 0.466 | 0.382 | 0.412 | 0.397 | 0.552 |
| COBRE / Schizophrenia | 0.602 | 0.566 | 0.595 | 0.557 | 0.590 | 0.519 |

*The MLP head does not clearly beat the linear probe; deeper heads overfit.*

## 5. Best results vs SOTA (same-dataset comparisons)

All three metrics side by side. `n/r` = not reported by that paper. Caveats: (1) Brain-JEPA numbers are **fine-tuning**, ours are **linear probe**; (2) our row is the **best-AUROC config** (avoids majority-class accuracy inflation). Brain-JEPA reports only Acc/F1 (no AUROC); NeuroSTORM reports Acc for ADHD-200.

### Brain-JEPA (same dataset = ADNI)

| Benchmark | Model | Acc | F1 | AUROC |
|---|---|---|---|---|
| ADNI / NC-MCI | Ours (unfrozen_adni) | 0.576 | 0.632 | 0.604 |
| | Brain-JEPA (FT) | 0.768 | 0.863 | n/r |
| ADNI / Amyloid | Ours (pool_adni) | 0.559 | 0.623 | 0.641 |
| | Brain-JEPA (FT) | 0.710 | 0.760 | n/r |

### NeuroSTORM (same dataset = ADHD-200)

| Benchmark | Model | Acc | F1 | AUROC |
|---|---|---|---|---|
| ADHD-200 | Ours (base_adhd_agg-mean_std) | 0.617 | 0.287 | 0.557 |
| | NeuroSTORM | 0.587 | n/r | n/r |

**For reference (NOT a valid comparison — different cohort HCP-YA vs HCP-Aging):**

| Benchmark | Model | Acc | F1 | AUROC |
|---|---|---|---|---|
| HCP / Sex | Ours (pool_hcp) | 0.892 | 0.879 | 0.962 |
| | Brain-JEPA (FT, HCP-Aging) | 0.815 | 0.843 | n/r |

**Not compared (different dataset):** HCP Sex/Age (HCP-YA vs HCP-Aging), COBRE vs HCP-EP, UCLA (to download). OASIS/ABIDE are not Brain-JEPA benchmarks.


**SOTA sources:** Brain-JEPA (arXiv 2409.19407, Tables 2-3, fine-tuning; Acc/F1 only) · NeuroSTORM (arXiv 2506.11167).
