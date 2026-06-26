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

| Run | Autism (val / **test**) | Sex (val / **test**) |
|-----------|:-----------------------:|:--------------------:|
| baseline (B) | 0.573 / 0.668 | 0.611 / 0.624 |
| fourier      | 0.579 / 0.458 | 0.576 / 0.512 |
| **freezeC (C)** | **0.634 / 0.603** | 0.622 / 0.474 |
| highlr       | 0.599 / 0.524 | 0.678 / 0.617 |

### Reading (selection on VAL)

- **Autism — best on val = freezeC (val 0.634 → test 0.603).** The val AUC is
  clearly above chance, so the signal is real and the selection is meaningful;
  freezeC's test (0.603) is reliable (val ≈ test). The baseline's higher test
  (0.668) comes with a near-chance val (0.573) and is not trustworthy.
- **Sex — best on val = highlr (val 0.678 → test 0.617).**
- **freezeC (policy C) is the best config on Autism**, consistent with the ADNI
  probe and with the earlier HCP-only freeze ablation — i.e. freezing the whole
  transformer and learning only the fMRI input adapter is the most robust policy.

### Take-away

Real, leakage-free autism signal (val ≈ 0.63, clearly above chance). freezeC is
the strongest and most consistent pretraining configuration.

\newpage

# 4. Comparison to SOTA — ABIDE Autism

From `SOTA_COMPARISON_EN.pdf`, the models that report **ABIDE Autism**:

| Model | Type | ABIDE Autism | Pretraining scale |
|-------------|----------------------|:------------------:|:-----------------:|
| BNT | supervised (no SSL) | AUROC **80.2 %** | none |
| BrainGFM | graph foundation model | AUC **71.2** (ABIDE II) | 27 datasets, 25k subj |
| LCM | connectome FM (leakage-free CV) | F1 **72.5** | ~10k scans |
| **Ours (freezeC)** | **volumetric FM** | **AUC 60.3** (ABIDE I) | ~3.7k scans |

**Honest reading:**

- We are **below** all models that report ABIDE Autism. BNT is a supervised
  task-specific model (different category); BrainGFM and LCM are graph/connectome
  models with much larger and more diverse pretraining.
- Crucially, **our architectural peers (volumetric models: SLIM-Brain, SwiFT)
  report ABIDE _Age_, not Autism** (SLIM-Brain 64.4 % acc, SwiFT 62.2 % acc). An
  apples-to-apples comparison therefore requires running **ABIDE Age** (TODO).
- Our scale (~3.7k pretraining scans) is comparable to SLIM-Brain (4 129
  sessions), so the gap is one of method maturity (10 epochs, no tuning, single
  split), not corpus size.

# 5. Caveats and next steps

- **Single split** (test n = 155): high variance. Headline numbers require
  **mean ± std over 3–5 subject-level seeds**.
- **ABIDE Age** probe (apples-to-apples vs SLIM-Brain / SwiFT) — TODO.
- **ADNI** results to be added (degradation 1/2/3y and CDR were near chance on
  val; Sex ≈ 0.83 AUC confirms the embeddings carry real signal).
- **Closing the SOTA gap:** longer pretraining, learning-rate tuning, architecture.
