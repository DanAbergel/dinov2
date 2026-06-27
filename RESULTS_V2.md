---
title: "Multi-source DINOv2 fMRI foundation model — V2 results"
date: "2026-06-26"
geometry: margin=2cm
fontsize: 10pt
---

# Multi-source fMRI foundation model — V2 results

This document reports the first leakage-free results of the V2 multi-source
DINOv2 fMRI foundation model, and compares them to the SOTA reviewed in
`SOTA_COMPARISON_EN.pdf`. It starts with the shared setup, then the four
pretraining ablations, then the downstream results (ABIDE first).

\vspace{0.4cm}

# 1. Setup (shared by all runs)

**Corpus (4 627 scans, 5 sources):** HCP-YA 1 084, ABIDE I 1 102, OASIS-3 1 197,
AOMIC PIOP 434, ADNI 812. All scans are `(T, 45, 54, 45)` volumes.

**Temporal harmonization:** every scan is resampled online from its native TR to
a common **TR = 0.72 s** (`scipy.signal.resample_poly`), then a fixed window of
**T_fixed = 270 frames** (≈ 194 s) is cropped. Per-scan native TR: HCP 0.72,
ABIDE per-site, OASIS 2.2, AOMIC 0.75/2.0, ADNI 3.0.

**Architecture:** official DINOv2 (student–teacher self-distillation + iBOT
masked-patch), ViT-S/14 with 4 register tokens, initialised from the ImageNet
DINOv2 checkpoint (transformer blocks only; the 3D patch-embed + positional
encoding are fMRI-specific and random-init). fMRI patchify = `PatchEmbed3DPlus1D`
(hierarchical 3D-spatial + 1D-temporal); token grid 27 × 150 = 4 050.

**Augmentation:** masking-only — all crops are the full volume (no spatial zoom,
not meaningful for a fixed brain), with MAE-style per-token random masking
(replacing BeiT block masking, which is spatially incoherent on the flattened
token grid).

**Batching:** `ProportionalBatchSampler` — every batch of 16 has a fixed
composition (HCP 4 / ABIDE 4 / OASIS 4 / ADNI 3 / AOMIC 1). 10 epochs.

**Leakage-free split:** subjects are split 70/15/15 (train/val/test) per dataset.
The val+test subjects of the downstream datasets (ADNI, ABIDE, OASIS) are
**excluded from SSL pretraining** (921 scans held out), so the encoder never sees
the evaluation subjects — comparable to Brain-JEPA / SLIM-Brain protocols.

**Probe protocol:** one embedding per scan = mean CLS token over sliding
T_fixed windows. For each label: LogReg fit on TRAIN subjects, regularisation `C`
selected on VAL (AUC), final AUC reported on TEST. **Selection is on VAL only;
TEST is touched once.** A `val_auc` near 0.5 means the selection failed → the
corresponding `test_auc` is not trustworthy (chance-level variance).

\newpage

# 2. The four pretraining ablations

All four share the setup above; each changes **one factor** vs the baseline
(one-factor-at-a-time design).

| Run | Spatial pos | Freeze policy | Base LR | Isolates |
|-----------|-------------|------------------------|---------|----------------------|
| **baseline** | learned table | **B** — train patch-embed + blocks 9–11 + norm + heads (blocks 0–8 frozen) | 3e-4 | reference |
| **fourier** | **Fourier features** of 3D coords | B | 3e-4 | effect of Fourier positional encoding |
| **highlr** | learned table | B | **1e-3** | effect of a higher learning rate |
| **freezeC** | learned table | **C** — train patch-embed + heads only (all 12 transformer blocks frozen) | 3e-4 | effect of the freeze policy (C vs B) |

**Freeze policies** (which DINOv2-initialised weights adapt to fMRI):

- **B (`fmri_plus_last_3`)** — freeze transformer blocks 0–8; the patch-embed,
  the last 3 blocks (9–11) and the final norm are trainable (~23.3 M params).
- **C (`fmri_only`)** — freeze the entire transformer (blocks 0–11); only the new
  3D patch-embed is trainable (~11.5 M params). The DINOv2 features are kept
  intact and only an fMRI input adapter is learned.

(A full-finetune / no-freeze run, "A", is not in this first batch; an earlier
HCP-only ablation found C > A > B, see `FREEZE_ABLATION_RESULTS.pdf`.)

\newpage

# 3. Downstream results — ABIDE

ABIDE is the canonical fMRI foundation-model benchmark. Probe on the held-out
ABIDE test subjects (n_train = 724, n_test = 155; Autism positives = 71, Sex
positives = 135). AUC, leakage-free.

### ABIDE — val AUC / **TEST AUC**

Each cell is **val AUC / TEST AUC / TEST Acc**. The reported config per task is
the one with the best VAL AUC (leakage-free selection).

| Run | Autism | Age | Sex |
|-----------|:-----------------:|:-----------------:|:-----------------:|
| baseline (B) | 0.573 / 0.67 / 0.61 | 0.843 / 0.86 / 0.79 | 0.611 / 0.62 / 0.60 |
| fourier      | 0.579 / 0.46 / 0.48 | 0.780 / 0.75 / 0.70 | 0.576 / 0.51 / 0.62 |
| **freezeC (C)** | **0.634 / 0.60 / 0.55** | **0.890 / 0.82 / 0.76** | 0.622 / 0.47 / 0.69 |
| highlr       | 0.599 / 0.52 / 0.49 | 0.833 / 0.87 / 0.81 | **0.678 / 0.62 / 0.65** |

### Val-selected result per task

| Task | Best config (val) | TEST AUC / Acc |
|------|-------------------|:--------------:|
| **Autism** | freezeC (val 0.634) | **0.60 / 0.55** |
| **Age** | freezeC (val 0.890) | **0.82 / 0.76** |
| **Sex** | highlr (val 0.678) | **0.62 / 0.65** |

- **freezeC (policy C) is the best config** on the two clinical/demographic tasks
  (Autism, Age) — consistent with the earlier HCP-only freeze ablation. Freezing
  the whole transformer and learning only the fMRI input adapter is the most
  robust policy.
- Baseline shows higher *test* on Autism (0.67) but with a near-chance *val*
  (0.573) → not trustworthy. We report the val-selected config (freezeC).

\newpage

# 4. Comparison to SOTA

### ABIDE Autism (AUC) — vs graph/connectome models

| Model | Type | ABIDE Autism |
|-------------|----------------------|:------------------:|
| BNT | supervised (no SSL) | AUROC 0.80 |
| LCM | connectome FM (leakage-free CV) | F1 0.73 |
| BrainGFM | graph FM (25k subj) | AUC 0.71 (ABIDE II) |
| **Ours (freezeC)** | **volumetric FM (~3.7k)** | **AUC 0.60** (ABIDE I) |

We are **below** the models that report ABIDE Autism. These are graph/connectome
models with much larger pretraining; no volumetric peer reports ABIDE Autism.

### ABIDE Age (Acc) — vs our volumetric peers

| Model | Type | ABIDE Age (Acc) |
|-------------|----------------------|:------------------:|
| SwiFT | volumetric FM | 62.2 % |
| SLIM-Brain | volumetric FM (4.1k) | 64.4 % |
| **Ours (freezeC)** | **volumetric FM (~3.7k)** | **76 %** |

We are **above** our direct architectural peers (SLIM-Brain, SwiFT) on ABIDE Age.
Caveat: our age binarization (split at the median) may differ from theirs, and
age is an easy target; this is a positive but not a strictly identical-protocol
comparison.

### Reading

The foundation model learns strong **demographic** structure (Age 0.82 AUC /
0.76 Acc, above peers) but the harder **clinical** task (Autism 0.60 AUC) is
below the larger SOTA models. At a pretraining scale comparable to SLIM-Brain
(~3.7k vs 4.1k), the Autism gap is one of method maturity (10 epochs, no tuning,
single split), not corpus size.

# 5. ADNI — not yet comparable (in progress)

Sagi's ADNI cohort is small (215 subjects). With a 70/15/15 split the test set is
~32 subjects (37–61 scans/label) — too small: the fixed-split probe **overfits the
val** (e.g. NC/MCI val 0.81 → test 0.52), so neither val nor test is reliable.
**Exception: ADNI Sex** (n≈125, val 0.76 → test 0.79, consistent) — a solid sanity
result confirming the embeddings carry real signal.

**Fix in progress:** a freezeC run with **ADNI fully excluded from pretraining**
(`train_v2_noadni`), then a subject-aware **k-fold probe over the full 215-subject
ADNI cohort** (encoder unseen) → a stable, SOTA-scale ADNI NC/MCI number
(vs Brain-JEPA 0.77, BNT 0.79 acc).

# 6. Caveats and next steps

- **Single split** (ABIDE test n = 155): high variance → headline numbers need
  **mean ± std over 3–5 subject-level seeds**.
- **ADNI k-fold** (full cohort, leakage-free) — running.
- **Closing the Autism gap:** longer pretraining, LR tuning, architecture.
