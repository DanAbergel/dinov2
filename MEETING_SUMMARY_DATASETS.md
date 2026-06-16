---
title: "Multi-source fMRI corpus — progress & open decisions"
date: "2026-06-16"
geometry: margin=2cm
fontsize: 10pt
---

# Goal

Build a multi-source resting-state fMRI corpus to pretrain the DINOv2-based foundation model, then evaluate downstream on Alzheimer's disease (ADNI). Strategy: a diverse multi-source corpus biased toward aging populations (as in SLIM-Brain, arXiv 2512.21881).

# 1. Datasets — status and properties

All scans are spatially downsampled to **45 × 54 × 45** voxels via trilinear interpolation, saved as tensors of shape `(T, 45, 54, 45)`. The temporal dimension `T` and the TR are not yet harmonized (see Problem 1).

| Dataset       | Scans | Population                | Age range | TR (s)   | Native T  | Access          |
|---------------|:-----:|---------------------------|-----------|:--------:|:---------:|-----------------|
| **HCP-YA**    | 1,206 | Young healthy adults      | 22–35     | 0.72     | 1200      | Open (S3)       |
| **ABIDE I**   | 1,102 | Autism + controls         | 6–64      | 1.5–3.0\* | 116–296   | Open (S3, PCP)  |
| **OASIS-3**   | 1,197 | Aging + AD (longitudinal) | 42–95     | 2.2      | ~164      | NITRC (XNAT)    |
| **AOMIC PIOP**| 442   | Young adults (Amsterdam)  | 18–26     | 0.75     | ~480      | Open (OpenNeuro)|
| **ADNI**      | 812   | Aging + MCI + AD          | 55–95     | 3.0      | ~140      | Sagi (preproc.) |
| **Total**     | **4,759** |                       |           |          |           |                 |

\* ABIDE TR varies by acquisition site (17 sites, e.g. NYU 2.0s, UCLA 3.0s, Pitt 1.5s).

**Total: 4,759 scans** — comparable in scale to SLIM-Brain (4,129), with a stronger aging bias (OASIS-3 + ADNI) for the Alzheimer's downstream task.

## Per-dataset notes

- **HCP-YA**: trilinear spatial downsampling (the previous version used every-other-voxel subsampling, which causes aliasing). 1 resting run per subject (REST1_LR).
- **ABIDE I**: Preprocessed Connectomes Project, CPAC pipeline, `nofilt_noglobal` strategy (no band-pass, no global signal regression — minimal, like the others). Already motion-corrected and MNI-registered.
- **OASIS-3**: 1 resting run per subject (longest run of the earliest session with rs-fMRI; a short ~5-frame calibration run is skipped).
- **AOMIC PIOP1+PIOP2**: fMRIPrep MNI rest BOLD; adds scanner diversity (Philips, Amsterdam).
- **ADNI**: no public S3 (LONI IDA only, raw scans in subject space). Two sources under evaluation — fresh from LONI, or Sagi's already-preprocessed version (see Problem 2).

# 2. Open problem 1 — heterogeneous TR across datasets

## The problem

Each dataset was acquired with a different repetition time (TR), ranging from **0.72s (HCP) to 3.0s (ADNI)**. A fixed temporal patch (e.g. 20 frames) therefore spans a **different physical duration** depending on the dataset (14.4s at TR 0.72s vs. 60s at TR 3.0s). Mixing datasets without correction means the model sees temporally inconsistent inputs.

## Constraints we imposed

1. **No artificial data.** We refuse to upsample (temporal interpolation that invents frames that were never acquired). Only downsampling (true frame decimation) is allowed.
2. **Proportional batches.** Each training batch must contain a fixed proportion of each dataset (for balanced multi-source learning). This requires a **homogeneous batch tensor** — all samples in a batch must have the same shape.

## Decision

**Resample every dataset to TR = 3.0s** (the maximum TR in the corpus, ADNI), by temporal decimation only.

| Dataset | Native T | T after resample to 3.0s |
|---------|:--------:|:------------------------:|
| HCP     | 1200     | 288                      |
| AOMIC   | ~480     | ~120                     |
| OASIS   | ~164     | ~120                     |
| ABIDE   | ~200     | ~133                     |
| ADNI    | ~140     | ~140 (unchanged)         |

## Why this decision

- **Respects constraint 1 (no invention):** resampling to the *maximum* TR means every dataset is only downsampled or left unchanged — never upsampled. No interpolated frames.
- **Respects constraint 2 (proportional batches):** a common TR lets us crop a fixed-length temporal window identical for all datasets, producing a homogeneous batch tensor.
- **Sufficient for resting-state connectivity:** functional connectivity lives in the 0.01–0.1 Hz band. By Nyquist, TR = 3.0s captures up to 0.17 Hz — still above the connectivity band. The higher frequencies lost from HCP (TR 0.72s) are mostly physiological noise (respiration ~0.3 Hz, cardiac ~1 Hz), routinely filtered out anyway.

## Trade-off acknowledged

HCP loses temporal resolution (1200 → 288 frames). This is a **real loss** (decimation), but not invented data. We accept it to keep the corpus consistent and the batches homogeneous. The alternative (a per-dataset temporal kernel that preserves all data) was rejected because it produces variable token counts and breaks the proportional-batch requirement.

## Rejected alternatives

- **TR = 0.72s (like SLIM-Brain):** would require upsampling all slower datasets — invents data, and inflates storage ~3× (e.g. OASIS 85 GB → 255 GB). Rejected.
- **Per-dataset temporal kernel (1 token = same physical duration regardless of TR):** preserves all data with no loss and no invention, but each dataset yields a different number of temporal tokens, making homogeneous proportional batches impossible. Rejected for now.

## Next step

We will measure the **distribution of temporal lengths** after resampling, then choose the fixed window size `T_fixed` (≤ the minimum length, so the window fits every scan without padding). A short scan can cap `T_fixed` for everyone, so we may exclude the few shortest scans to keep a usable window. This will be decided from the real length histogram, not guessed.

# 3. Open problem 2 — ADNI preprocessing consistency

## The problem

The other four datasets (HCP, ABIDE, OASIS, AOMIC) are all in **minimal preprocessing** (motion correction + MNI registration, no heavy denoising). For ADNI we have two sources:

- **Fresh from LONI IDA:** raw DICOM in subject space — requires a full pipeline (dcm2niix → motion correction → MNI registration). Weeks of work and compute.
- **From collaborator (Sagi):** 812 scans already preprocessed, but with **CONN** (heavy denoising: band-pass filtering, motion regression, possibly global signal regression).

## The risk

Mixing four "minimally preprocessed" datasets with one "heavily denoised" dataset (Sagi/CONN) could let the SSL model **distinguish ADNI by its noise level** rather than learning generalizable anatomical features — a site/batch bias.

## Decision (tentative)

For the **V1 pretraining**, use **Sagi's ADNI** (fast, labels already available for downstream evaluation), while keeping the fresh LONI download in reserve as a "clean" version to (a) measure whether the CONN preprocessing introduces a detectable bias and (b) use for the final thesis results if needed.

## Why

- ADNI fresh preprocessing would take weeks; Sagi's data unblocks the corpus immediately.
- We already have the downstream labels (CDR, MMSE, conversion) from the V1 work with Sagi's data.
- The bias risk is real but measurable — if the model separates ADNI too easily from the others, that is a clear warning sign.
- The fresh ADNI (currently downloading) gives us a fallback / validation set.

## Also decided for ADNI scan types

ADNI offers four rs-fMRI descriptions. We take only the **three single-band protocols** (`Resting State fMRI`, `Extended Resting State fMRI`, `Axial rsfMRI (Eyes Open)`) and **exclude the multi-band** (`Axial MB rsfMRI`, TR 0.6s, ~960 frames). Reasons: the single-band protocols form one consistent family (TR 3s), match Sagi's data and the rs-fMRI used by the ADNI SOTA comparators (Brain-JEPA, SLIM-Brain), whereas the multi-band is a fundamentally different acquisition and minority in the data.

# 4. Summary of decisions

| # | Decision | Reason |
|---|----------|--------|
| 1 | Spatial downsample to 45×54×45 (trilinear) | Common grid, anti-aliased |
| 2 | Resample all to TR = 3.0s (downsample only) | No invented data + homogeneous batches + sufficient for connectivity |
| 3 | Fixed temporal window `T_fixed` (TBD from length histogram) | Homogeneous batch tensor for proportional sampling |
| 4 | Proportional batch sampler | Balanced multi-source learning |
| 5 | ADNI: use Sagi's preprocessed data for V1, fresh LONI in reserve | Speed + existing labels; validate bias later |
| 6 | ADNI: single-band rs-fMRI only (exclude multi-band) | Protocol consistency with corpus and SOTA comparators |

# 5. Open questions for the meeting

1. **TR target = 3.0s:** acceptable to decimate HCP from 1200 to 288 frames, or should we keep a per-dataset kernel and find another way to handle proportional batches?
2. **ADNI source:** is Sagi's CONN-preprocessed data acceptable for V1 pretraining, or should we wait for the fresh minimally-preprocessed LONI download?
3. **Batch proportions:** equal across the five datasets, or weighted toward the aging cohorts (OASIS, ADNI) given the Alzheimer's downstream target?
