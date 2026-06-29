---
title: "Multi-source DINOv2 fMRI foundation model — v1 results"
date: "2026-06-29"
geometry: margin=2cm
fontsize: 10pt
---

# Multi-source fMRI foundation model — v1 results

This document reports the first complete, leakage-free results of the multi-source
DINOv2 fMRI foundation model. It is organised in four parts:

1. **What the 2026-06-14 meeting asked, and how each point was implemented**
   (including where the implementation deviated from the written plan, and why).
2. The **two ablations** that were run (base vs Fourier).
3. The **evaluation protocol** (leakage-free holdout, 70:30 split, cross-validation).
4. The **per-task results next to every SOTA that reports the same benchmark.**

\vspace{0.3cm}

## Executive summary (two messages)

- GOOD:  **Demographic / global structure is strong.** On **ABIDE Age** we reach
  **0.80 accuracy, above** the volumetric peers (SLIM-Brain 0.64, SwiFT 0.62);
  on **HCP Sex** we reach **0.81 acc / 0.79 F1, above LCM** (0.73 F1).
- WEAK:  **Fine clinical signal is near chance.** Autism (ABIDE) and early Alzheimer
  (ADNI NC/MCI, AD/HC) stay around 0.5–0.64, **below** the graph/connectome SOTA.
- The **Fourier** positional encoding **does not help** (base >= Fourier on every
  axis that carries signal) -> the retained model is **base**.

\newpage

# 1. From the meeting plan to the implementation

The 2026-06-14 meeting (`MEETING_SUMMARY_2026-06-14`) defined four work items.
Below is what was asked and what was actually built — two items were implemented
**differently** from the written plan, for concrete technical reasons.

## 1.1 Multi-source pretraining corpus

**Asked:** extend HCP-YA-only pretraining with aging cohorts (HCP-Aging, OASIS-3,
AOMIC), ~3.4k sessions, to close the age gap with the ADNI downstream target.

**Built:** a **5-source corpus of 4 627 scans**, all resampled to `(T, 45, 54, 45)`:

| Source | Scans | Role |
|--------|:-----:|------|
| HCP-YA | 1 084 | young healthy baseline |
| ABIDE I | 1 102 | autism + demographic benchmark |
| OASIS-3 | 1 197 | aging + AD |
| AOMIC PIOP1/2 | 434 | scanner diversity |
| ADNI | 812 | Alzheimer downstream target |

> **Deviation:** HCP-Aging access (NDA-controlled, 1–3 months) was not ready, so
> **ABIDE I and ADNI** were used instead — they also give us the canonical fMRI
> benchmarks (ABIDE autism, ADNI NC/MCI). Scale (~4.6k) matches the SLIM-Brain
> target.

## 1.2 Mixing scans of different length / different TR

**Asked (written plan):** pad every scan along time to the longest $T = 1200$
(HCP-YA), zero-padding the end; respect a padding mask in the loss.

**Built — a different solution (Ariel's later decision):** padding to 1200 would
have meant **60–85% zeros** for most datasets, which the masking objective cannot
learn from. Instead we do **online temporal harmonisation**:

- every scan is resampled from its native TR to a common **TR = 0.72 s** using
  `scipy.signal.resample_poly` (we crop the native window **first**, then resample
  — this keeps the polyphase filter cheap);
- then a fixed window of **$T_{\text{fixed}} = 270$ frames** (~ 194 s) is taken.

This resolves the meeting's **open question 2 (TR heterogeneity)** by option 1
(resample to a common TR): one token now spans the **same physical duration**
across all datasets. Native TRs: HCP 0.72, ABIDE per-site, OASIS 2.2, AOMIC
0.75/2.0, ADNI 3.0.

**Patchify** (unchanged `PatchEmbed3DPlus1D`): hierarchical 3D-spatial + 1D-temporal,
`temporal_kernel = 10` => $T_{\text{eff}} = 27$, token grid $27 \times 150 = 4050$.

## 1.3 Augmentation: masking-only

**Asked:** replace DINOv2 multi-crop (spatial sub-volumes + BeiT block masking) by
**1 global + 3 local** full-image crops with **per-token Gaussian masking**, because
(a) cropping a sub-volume of a fixed brain is not semantically meaningful, and
(b) BeiT block masking is spatially incoherent on the flattened token grid.

**Built — the masking part, pragmatically:** all crops are the **full volume**
(no spatial zoom) with **per-token random masking** (`RandomTokenMaskingGenerator`,
ratio sampled in [0.10, 0.50]), which replaces BeiT block masking as asked. We
**kept DINOv2's proven 2-global + 3-local self-distillation loss intact** rather
than rewriting it to the literal "1 global + 3 local" — the literal form would
require rewriting `forward_backward` and was judged too risky for a first
iteration. So: **full-image crops + MAE-style per-token masking, original loss.**

## 1.4 Fourier features for spatial position (the §3 ablation)

**What the model needs.** Each of the 150 spatial patches must be told *where it
sits* in the brain so the transformer can use spatial relationships. By default
this is a **learned lookup table** (`pos_spatial`, 150x384): 150 vectors learned
from scratch, with **no built-in geometry** — nothing tells the model that patch
(0,0,0) is a neighbour of (0,0,1).

**What Fourier features do (Tancik et al., NeurIPS 2020).** A plain network fed raw
coordinates suffers *spectral bias*: it cannot tell apart inputs that are **close
together** — it blurs nearby positions into the same output. Fourier features fix
this by mapping a coordinate through **cos/sin at many frequencies**: two nearby
positions then produce **clearly different** signatures (at high frequency) while
staying correlated (at low frequency). In short, they let the network **resolve
fine spatial distances** that raw coordinates cannot express.

**Built — exactly as the meeting's Option B,** in
`dinov2/layers/patch_embed_3d_plus_1d.py`. The encoding is computed in **three
steps** that replace the learned table:

1. **Coordinate** — give each of the 150 patches its 3D position in `[-1, 1]`
   (`_make_grid_coords`, in the same order the patchify flattens the tokens).
2. **Fourier map** — `gamma(v) = [cos(2*pi*Bv), sin(2*pi*Bv)]`: turn the 3 numbers
   into 64 (32 cosines + 32 sines), where `B` is a fixed random frequency matrix.
3. **MLP** — a 2-layer MLP maps the 64 features to `embed_dim = 384`, the per-token
   size, and this is added to each token as its spatial position.

```python
class FourierFeatures3D(nn.Module):       # step 2
    def __init__(self, num_freqs=32, sigma=10.0):
        self.register_buffer('B', torch.randn(num_freqs, 3) * sigma)  # FIXED freqs
    def forward(self, pos):                # (150,3) -> (150,64)
        proj = 2*math.pi * pos @ self.B.t()
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
```

A single switch (`get_spatial()`) returns either the learned table (**base**) or
this Fourier chain (**fourier**) — that switch is the only difference between the
two runs. Hyperparameters: **`num_freqs = 32`, `sigma = 10`, `B` fixed** (the
meeting's "fixed vs learnable B" -> fixed for v1). Only the **spatial** position is
replaced; temporal and CLS positions stay learned. Toggled by
`student.fmri_fourier_pos` (`FOURIER=1`).

## 1.5 Transfer learning + freeze policy

The transformer is **initialised from the ImageNet DINOv2 checkpoint** (ViT-S/14,
4 register tokens); only the fMRI-specific 3D patchify + positional encoding are
random-init. We use freeze policy **B (`fmri_plus_last_3`)**: transformer blocks
0–8 are frozen, and the patchify + blocks 9–11 + final norm + heads are trained
(~23.3 M trainable params). Both ablations use policy B.

## 1.6 Leakage (meeting open question 1) — resolved

The meeting left "can we pretrain on the full data without leakage" open. **Resolved
(2026-06-28, confirmed by Yoni):** exclude the **downstream test subjects from SSL
pretraining**. `HOLDOUT_DATASETS = (ADNI, ABIDE, OASIS, HCP)` — the held-out 30% of
each is removed from pretraining, so the encoder never sees an evaluation subject.
This matches the LCM / Brain-JEPA practice. (Full SOTA-exact alignment would
pretrain on a disjoint UK Biobank — planned for v2 once access is granted.)

\newpage

# 2. The two ablations

Both runs share everything in Section 1 and differ by **one factor only**
(one-factor-at-a-time design): the spatial positional encoding.

| Run | Spatial position | Freeze | Init | Epochs |
|-----|------------------|--------|------|:------:|
| **base** | learned table | B (`fmri_plus_last_3`) | DINOv2 ImageNet | 10 |
| **fourier** | **Fourier features** of 3D coords | B | DINOv2 ImageNet | 10 |

Each run: ~23 000 iterations, effective batch 16 (micro-batch 2 x grad-accum 8),
proportional sampling across the 5 sources, single GPU (~3.5 h).

\newpage

# 3. Evaluation protocol

**Embedding.** For each scan: one embedding = mean CLS token over sliding
$T_{\text{fixed}}$ windows (resampled to 0.72 s), from the **teacher** encoder.

**Split — 70:30, subject-level.** Subjects (not scans) are split 70/30: a subject's
scans are near-duplicates, so a scan-level split would leak identity. **TRAIN = 70 %
of subjects; TEST = the held-out 30 %** (these were excluded from pretraining).

**Choosing the classifier — cross-validation on TRAIN, no separate val set.** The
probe is an L2-regularised logistic regression; its regularisation `C` is selected
by **subject-aware k-fold cross-validation on the TRAIN portion only** (folds are
grouped by subject). The 30 % test set is **never** touched during selection. This
is more robust than a fixed small val set and mirrors the LCM cross-validation
protocol.

**Metrics — matched to each SOTA.** Most SOTA papers report **Accuracy and F1, not
AUC**. We therefore report, **per task, the metric the competitor reports**: AUROC
for ABIDE-Autism (vs BNT), F1 for the LCM comparisons, Accuracy for ABIDE-Age and
ADNI-NC/MCI (vs SLIM-Brain / Brain-JEPA), etc. F1 is binary (positive class), as in
the SOTA.

> **Honesty note.** A near-chance task can still show an inflated F1 under class
> imbalance + `class_weight=balanced` (e.g. ADNI NC/MCI F1 0.66 while Acc 0.55) —
> for those we read the **Accuracy / AUC** as the truthful number.

\newpage

# 4. Results per task vs SOTA

Each block lists **every SOTA that reports the same benchmark**, with **its own
metric**, next to our two runs. Bold = our best run on that line.

## ABIDE — Autism (psychiatric, the canonical fMRI-FM benchmark)

| Model | Type | Metric | Score |
|-------|------|--------|:-----:|
| BNT | supervised connectome | AUROC | 0.80 |
| BrainGFM | graph FM (25k subj) | AUC | 0.71 (ABIDE II) |
| LCM | connectome FM | F1 | 0.73 |
| **Ours base** | volumetric FM (~4.6k) | AUROC / F1 | **0.53 / 0.50** |
| Ours fourier | volumetric FM | AUROC / F1 | 0.50 / 0.52 |

-> **Near chance, well below SOTA.** No volumetric peer reports ABIDE Autism; the
models that do are larger graph/connectome models.

## ABIDE — Age (demographic)

| Model | Type | Metric | Score |
|-------|------|--------|:-----:|
| SwiFT | volumetric FM | Acc | 0.622 |
| SLIM-Brain | volumetric FM (4.1k) | Acc | 0.644 |
| **Ours base** | volumetric FM (~4.6k) | Acc | **0.80** |
| Ours fourier | volumetric FM | Acc | 0.71 |

-> **Above our direct volumetric peers.** (Caveat: our age binarisation is at the
median; theirs may differ — positive but not strictly identical protocol.)

## ABIDE — Sex (sanity)

| Model | Metric | Score |
|-------|--------|:-----:|
| LCM | F1 | 0.873 |
| **Ours base / fourier** | F1 | **0.82 / 0.82** |

-> Close to LCM. (ABIDE Sex is class-imbalanced, so we read F1 as the SOTA does.)

## ADNI — NC vs MCI (early Alzheimer, the main clinical duel)

| Model | Type | Metric | Score |
|-------|------|--------|:-----:|
| BNT | supervised | Acc | 0.789 |
| Brain-JEPA | UKB-pretrained FM | Acc / F1 | 0.768 / 0.863 |
| SLIM-Brain | volumetric FM | Acc / F1 | 0.691 / 0.690 |
| SwiFT | volumetric FM | Acc | 0.645 |
| **Ours base** | volumetric FM | Acc / F1 | **0.55 / 0.66** |
| Ours fourier | volumetric FM | Acc / F1 | 0.54 / 0.65 |

-> **Near chance (Acc 0.55).** The F1 0.66 is inflated by imbalance — the Accuracy
is the honest number.

## ADNI — AD vs HC (full Alzheimer)

| Model | Type | Metric | Score |
|-------|------|--------|:-----:|
| OViTAD | supervised 2D ViT | F1 | 0.99 (n=31 test) |
| BrainGFM | graph FM | AUC / Acc | 0.803 / 0.851 |
| LCM | connectome FM | F1 | 0.853 |
| Ours base | volumetric FM | AUC / Acc / F1 | 0.64 / 0.54 / 0.50 |
| **Ours fourier** | volumetric FM | AUC / Acc / F1 | 0.64 / **0.61 / 0.54** |

-> **Modest (AUC 0.64), below SOTA.**

## HCP — Sex (demographic)

| Model | Type | Metric | Score |
|-------|------|--------|:-----:|
| SLIM-Brain | volumetric FM | Acc / F1 | 0.911 / 0.911 |
| LCM (HCP-YA) | connectome FM | F1 | 0.722 |
| **Ours base** | volumetric FM | Acc / F1 | **0.81 / 0.79** |
| Ours fourier | volumetric FM | Acc / F1 | 0.79 / 0.77 |

-> **Above LCM (F1 0.79 vs 0.72), below SLIM-Brain.**

\newpage

# 5. Reading

The foundation model learns **global / demographic** brain structure well — it
**beats the volumetric SOTA on ABIDE Age and beats LCM on HCP Sex** — but the
**fine clinical signal** (autism, early Alzheimer) stays near chance, below the
graph/connectome models. This is consistent with the training behaviour we
diagnosed: with a mostly-frozen backbone, masking-only augmentation and 10 epochs,
the encoder captures coarse structure but not subtle diagnostic patterns.

**Fourier vs learned position:** Fourier **did not help** — base is >= Fourier on
every axis that carries signal (notably Age 0.80 vs 0.71, HCP Sex 0.81 vs 0.79).
**Why this makes sense:** Fourier features pay off when coordinates are **dense and
continuous** (e.g. NeRF), so the network must resolve points only a tiny distance
apart. Our token grid is the opposite — only **150 discrete patches**, spaced ~18 mm
apart and well separated. The "distinguish closely-spaced positions" problem Fourier
is designed for is **barely present here**, and a 150-entry learned table already has
ample capacity to place those few positions. Fourier would become relevant at a
**finer resolution** (smaller patches, far more positions, or continuous voxel
coordinates) — a v2 lever. The clean ablation conclusion for v1: **keep the learned
positional table.**

# 6. Limitations and v2

- **Single 70:30 split** — headline numbers should get **mean +/- std over 3–5
  subject-level seeds** for publication.
- **Clinical gap** — the autism / Alzheimer gap is the target of v2.
- **v2 direction (decided):** to be **directly comparable to Brain-JEPA / BrainLM**,
  pretrain on a **large disjoint corpus (UK Biobank)** and evaluate on the small
  datasets fully held out by construction. UK Biobank access (DUA) is being
  applied for (2–6 months); v1 is the interim deliverable.

# Cited SOTA

Brain-JEPA (NeurIPS 2024), BrainLM (ICLR 2024), SLIM-Brain (preprint 2025),
BrainGFM (preprint 2025), LCM (AAAI 2026), BNT (NeurIPS 2022), SwiFT (NeurIPS 2023),
OViTAD (Brain Sciences 2023). Full details in `SOTA_COMPARISON_EN`.
