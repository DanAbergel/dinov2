# fMRI Foundation Model (V2) — Architecture (detailed)

A DINOv2 self-supervised foundation model on a multi-source fMRI corpus, evaluated
by a leakage-free linear/MLP probe. Goal: pretrain a transferable encoder;
downstream thesis target = ADNI progression.

---

## 0. Data pipeline (identical in training and probe)

**Spatial.** Every scan is downsampled offline to a fixed **(45, 54, 45)** volume.

**Temporal TR harmonization.** Native TR differs per dataset:
HCP 0.72 s · ABIDE **per-site** (1.5–3.0 s) · OASIS 2.2 s · AOMIC 0.75/2.0 s · ADNI 3.0 s.
To mix them we harmonize to a common **TARGET_TR = 0.72 s** and a fixed window
**T = 270 frames** (= 194.4 s of brain activity):
1. native window length `win = round(270 × 0.72 / TR_native)` frames — spans 194.4 s.
2. crop that window (on the mmap, so a 500 MB HCP scan never fully loads).
3. resample native→270 frames with **`scipy.signal.resample_poly`** (polyphase,
   anti-aliased FIR — the correct tool for band-limited BOLD).

**Normalization.** **Per-frame spatial z-score**: for each timepoint, subtract the
mean over voxels and divide by the std over voxels → each 3D volume is zero-mean /
unit-std. *(We do NOT z-score per-voxel over time — a candidate ablation.)*

Output of the pipeline: a **(270, 1, 45, 54, 45)** tensor.

---

## 1. Pretraining (self-supervised)

**Objective — DINOv2.** Self-distillation between a **student** and a **teacher**
(the teacher = EMA of the student). Losses:
- **DINO**: cross-entropy student↔teacher on the [CLS] token (with centering + sharpening),
- **iBOT**: masked-token prediction,
- **KoLeo**: feature-spreading regularizer.

**Patchify — `PatchEmbed3DPlus1D`** (turns the 4D scan into tokens):
- **Spatial** Conv3d, kernel/stride **9** → `5 × 6 × 5 = 150` spatial tokens.
- **Temporal** Conv1d, kernel **10** → `270 / 10 = 27` temporal tokens.
- → token grid **27 × 150 = 4050 tokens**, each embedded to **384-d**.

**Backbone.** **ViT-Small** (embed 384), reg4, **initialized from the DINOv2
ViT-S/14 ImageNet checkpoint** (patch_embed + pos_embed dropped — shapes differ —
and re-learned on fMRI).

**Augmentation — masking-only.** Crops = the **full volume** (2 global + N local,
no spatial crop: a brain is a fixed anatomical structure); the only corruption is
**per-token random masking** (MAE-style) applied in the collate for iBOT.

**Batch composition — `ProportionalInfiniteSampler`.** Fixed per-batch quota
**HCP 4 / ABIDE 4 / OASIS 4 / ADNI 3 / AOMIC 1 = 16**.

**Corpus.** 5 sources, **4627 scans**. Subject-level **holdout**: the 30% val+test
subjects of HCP/ABIDE/OASIS/ADNI are **excluded from pretraining** (leakage-free
probe); AOMIC is kept whole.

**Freeze policy (base).** Freeze DINOv2 blocks 0–8; train **patch_embed + blocks
9–11 + norm + heads** (`fmri_plus_last_3`).

**Optimization.** base_lr 3e-4, warmup 0.3 ep, **10 epochs**, patch_embed_lr_mult
0.2, effective batch 16 (micro-batch 2 × grad_accum 8).

**The 5 ablation runs** (one factor each vs base):
`base` (reference) · `fourier` (Fourier positional encoding) · `noblock2` (drop a
patchify block) · `pool` (AvgPool downsampling vs strided conv) · `unfrozen` (all
transformer layers trainable during SSL).

---

## 2. Probe (evaluation — encoder FROZEN)

**Embedding extraction.** Same spatial / TR / normalization pipeline as training.
Per scan: slide 270-frame windows (stride = win/2), run each through the frozen
**teacher**, take the **[CLS] token** per window, and **average** → one **384-d
vector per scan** (cached to disk).
*Difference vs training: training samples ONE random 270-window per step; the probe
averages ALL windows for a stable, deterministic embedding.*

**Probe head — `Probe(nn.Module)`, PyTorch.**
`hidden=()` → single Linear (384→1) = logistic regression; `hidden=(256,128)` →
MLP (Linear→ReLU→…→1). Trained with **Adam (lr 1e-3, wd 1e-4)** +
**BCEWithLogitsLoss** (class-balanced `pos_weight`) + **backprop**, 200 epochs.
Features standardized on the train split. **The encoder stays frozen** — only the
head is trained.

**Split — 70:30 leakage-free.** In-corpus datasets use `subject_split.json` (the
30% test was held out of pretraining); external datasets (ADHD-200/COBRE/UCLA,
never pretrained on) get a deterministic random 70:30.

**Metrics.** **AUROC** to rank the pretraining runs (threshold- and
balance-independent); **Acc / F1** to match what the SOTA papers report.

**Probed datasets.** ADNI · ABIDE · HCP · OASIS (in-corpus, held-out 30%) +
ADHD-200 · COBRE · UCLA (external, fully unseen).

---

## 3. Key points (honest)

- Strong **demographic** signal (HCP Sex AUROC 0.96); **clinical weak with a linear
  probe** → **fine-tuning** is the decisive next experiment.
- Same-dataset SOTA comparisons: **ADNI** (Brain-JEPA) and **ADHD-200** (NeuroSTORM).
  Brain-JEPA numbers are fine-tuning; ours are linear probe.
- Open questions: per-voxel temporal normalization (ablation), finishing the
  fine-tune, adding UCLA / HCP task-fMRI.
