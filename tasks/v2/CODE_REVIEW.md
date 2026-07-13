# Code review — our fMRI changes on top of DINOv2

Every change we made to the official DINOv2 (`facebookresearch/dinov2`), traced in
the **order the data flows through the pipeline**. Tags: 🟢 **new file**, 🟠 **modified
official file**. Each item = **What** / **Why** (vs official) / **Where** (`file : symbol`, line).

---

## Phase 1 — Corpus construction (before any sample is read)

**1.1 · Corpus constants** 🟢 `dinov2/data/fmri_const.py`
- **What**: the 5-dataset list, each dataset's native TR, the harmonization target
  (TR = 0.72 s), the fixed window T = 270, the holdout list, manifest/split filenames.
- **Why**: DINOv2 trains on one dataset (ImageNet); we mix 5 fMRI cohorts with
  different TRs, so the corpus must be declared somewhere.
- **Where**: `TARGET_TR` L11 · `DEFAULT_T_FIXED` L17 · `CORPUS_DATASETS` L22 · `HOLDOUT_DATASETS` L27.

**1.2 · Scan discovery (glob) + offline manifest** 🟢 `dinov2/data/fmri_data.py`
- **What**: walk each dataset dir, attach the per-scan native TR; the offline manifest
  records every scan's native length T so we can drop too-short scans without re-reading them.
- **Why**: no such multi-source discovery exists in DINOv2.
- **Where**: `fmri_data.py`: `DATASET_SOURCES` (declarative per-dataset table) · `build_corpus_entries` ·
  **offline** manifest builder in `dinov2/data/fmri_offline.py : write_corpus_manifest`
  (kept out of the training module — it runs once, not during training).

**1.3 · Subject-level holdout, 70/30 train/test (no leakage)** 🟠 `dinov2/data/fmri_data.py`
- **What**: a subject-level **70/30 train/test** split (`subject_split.json`); the 30 %
  **test subjects** of the downstream datasets are excluded from pretraining, so the
  probe later evaluates on subjects the encoder never saw.
- **Why**: leakage-free evaluation; DINOv2 has no notion of downstream holdout. We split
  by SUBJECT (a subject's scans never straddle train/test). No separate `val`: we never
  tuned on a held-out val set, so it was redundant (`train` = 70 %, `test` = the rest).
- **Where**: `entries_from_manifest` L103 (holdout filter) · `_load_split_map` L96 ·
  split built by `tasks/data_prep/make_subject_split/make_subject_split.py`.

**1.4 · The training dataset** 🟢 `dinov2/data/fmri_data.py`
- **What**: one `Dataset` presenting the 5 sources as one; exposes `dataset_indices`
  (name → global indices) for the sampler.
- **Why**: replaces the ImageNet dataset.
- **Where**: `MixedFMRIDataset` L209.

**1.5 · Per-batch dataset quota (proportional sampling)** 🟢 + 🟠
- **What**: every batch has a fixed composition — HCP 4 / ABIDE 4 / OASIS 4 / ADNI 3 /
  AOMIC 1 = 16 — instead of uniform sampling.
- **Why**: the sources differ hugely in size; without a quota HCP dominates the batch.
- **Where**: `fmri_data.py : ProportionalBatchSampler` L275 🟢 ·
  `data/loaders.py` L84 (`name=="Mixed"`), L34/L137 (`SamplerType.PROPORTIONAL`) 🟠 ·
  `data/samplers.py : ProportionalInfiniteSampler` L232 (DDP) 🟠.

---

## Phase 2 — Loading one sample (`__getitem__`)

**2.1 · Entry point** 🟢 `dinov2/data/fmri_data.py`
- **What**: `__getitem__` calls `_load(idx)` then applies the transform (augmentation).
- **Where**: `MixedFMRIDataset.__getitem__` L265 · `_load` L259.

**2.2 · Lazy memory-map** 🟢 `dinov2/data/fmri_data.py`
- **What**: the scan is memory-mapped, so a 500 MB HCP file never fully loads into RAM.
- **Why**: memory efficiency across 5 large cohorts.
- **Where**: `_load_mmap` L162.

**2.3 · Native temporal window (random crop)** 🟢 `dinov2/data/fmri_data.py`
- **What**: pick a native window of `win = round(270 × 0.72 / TR_native)` frames at a
  **random start**; crop it on the mmap (crop first → materialize only the window).
- **Why**: fixed real-time coverage (194.4 s) regardless of TR; random start = temporal
  augmentation at training.
- **Where**: `_native_window` L167 (used in `_load` L259).

---

## Phase 3 — Upsampling / TR harmonization (`_finalize`)

**3.1 · Spatial resize to (45, 54, 45)** 🟠 `dinov2/data/fmri_data.py`
- **What**: trilinear resize of the window to the fixed spatial grid (only if needed).
- **Why**: all sources must share one spatial shape to batch together.
- **Where**: `_finalize` L195 · `F.interpolate(...)` L201.

**3.2 · Temporal resample to 270 frames @ 0.72 s** 🟢 `dinov2/data/fmri_data.py`
- **What**: resample the native window (native TR) to exactly 270 frames at TR 0.72 s,
  using **polyphase / anti-aliased FIR** (`scipy.signal.resample_poly`).
- **Why**: harmonize heterogeneous TRs to a common rate; polyphase is the correct,
  anti-aliased tool for a band-limited BOLD signal (per Ariel).
- **Where**: `_temporal_resample` L177 · `resample_poly(...)` L185.

---

## Phase 4 — Normalization

**4.1 · Per-frame spatial z-score** 🟢 `dinov2/data/fmri_data.py`
- **What**: each timepoint volume is z-scored across voxels (subtract mean, divide by std).
- **Why**: put every frame on a common scale before the encoder. *(Note: this is
  spatial, not per-voxel temporal — a candidate ablation.)*
- **Where**: `_zscore_per_frame` L155 (called at the end of `_finalize`).

---

## Phase 5 — Augmentation (masking-only)

**5.1 · Full-volume "multi-crop"** 🟢 `dinov2/data/fmri_data.py`
- **What**: the DINO views are the **full volume** repeated (2 global + N local), i.e.
  NO spatial crop; the only corruption is per-token masking.
- **Why**: a brain is a fixed anatomical structure, not a scene to crop (meeting §2).
- **Where**: `MaskingAugmentation3D` L327 · `__call__` L341.

**5.2 · Per-token random masking (for iBOT)** 🟢 `dinov2/data/masking.py`
- **What**: MAE-style random token masking over the flattened (T_eff × N_spatial) grid,
  instead of DINOv2's 2D BeiT block masking.
- **Why**: the fMRI token order doesn't respect 2D neighborhoods, so block masking
  doesn't apply.
- **Where**: `RandomTokenMaskingGenerator` L11.

**5.3 · Mask-generator wiring for fMRI** 🟠 `dinov2/train/train.py`
- **What**: compute `n_tokens` and pick the mask generator from the fMRI token grid
  (T_eff × N_spatial) rather than the official 2D `(img/p, img/p)`.
- **Where**: `do_train` L253 (fmri branch).

---

## Phase 6 — Embedding: the 3D+1D patchify

**6.1 · Conv3Plus1d — the atomic op** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: a factorized conv = **3D spatial conv** then **1D temporal conv** (the fMRI
  analogue of MovieGen's Conv2Plus1d).
- **Why**: separates spatial and temporal patchification cheaply.
- **Where**: `Conv3Plus1d` L31.

**6.2 · Hierarchical patchify** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: `conv_in → block_0 → down_0 → block_1 → down_1 → block_2` turns
  (B, 270, 1, 45, 54, 45) into a token grid **27 (temporal) × 150 (spatial) = 4050**
  tokens of dim 384. `_ResBlock3Plus1d` = two Conv3Plus1d.
- **Why**: hierarchical spatial+temporal downsampling into transformer tokens.
- **Where**: `PatchEmbed3DPlus1D` L181 · `conv_in` L227 · `block_0` L233 · `down_0` L236 ·
  `_ResBlock3Plus1d` L70.

**6.3 · Architecture ablation flags** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: `remove_block2` (drop the last ResBlock) and `pool_downsample` (do all
  downsampling with AvgPool instead of strided convs).
- **Why**: the point-2 ablations (noblock2 / pool runs).
- **Where**: `PatchEmbed3DPlus1D.__init__` — `remove_block2` L201 · `pool_downsample` L202.

**6.4 · Positional embedding (learned or Fourier)** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: spatial positions come from a learned table (default) or **Fourier features**
  of 3D coordinates (the `fourier` ablation).
- **Why**: give the transformer spatial position; test Fourier vs learned.
- **Where**: `PositionEmbedding3D` L106 · `FourierFeatures3D` L87.

**6.5 · Bind the patchify into the model** 🟠 `dinov2/models/__init__.py`
- **What**: when `cfg.student.fmri_mode` is set, build the `PatchEmbed3DPlus1D` (with all
  fMRI flags) and pass it as the ViT's `embed_layer`, and set `img_size` from the fMRI dims.
- **Why**: swap the official 2D `PatchEmbed` (Conv2d) for our 3D+1D one.
- **Where**: `build_model_from_cfg` — embed_layer forwarding L32 · fmri branch L54–L74.

**6.6 · ViT accepts 6D fMRI input** 🟠 `dinov2/models/vision_transformer.py`
- **What**: an early branch handles input shaped (B, T, C, X, Y, Z) and uses the
  factorized positional embedding carried by `PatchEmbed3DPlus1D` instead of the flat
  `self.pos_embed`.
- **Why**: the official ViT expects 2D images; fMRI is 6D.
- **Where**: `prepare_tokens...` L217 (6D branch) · factorized pos L234.

---

## Phase 7 — DINO architecture / SSL training

**7.1 · Freeze policy (partial fine-tune)** 🟠 `dinov2/train/train.py`
- **What**: freeze part of the ImageNet-init transformer during SSL —
  `fmri_only` (train only patch_embed), `fmri_plus_last_3` (patch_embed + blocks 9–11 + norm),
  or none.
- **Why**: transfer from ImageNet while adapting the fMRI-specific parts; the freeze
  ablations (base vs unfrozen).
- **Where**: `apply_freeze_policy` L130 (`fmri_only` L158, `fmri_plus_last_3` L160),
  applied at L221.

**7.2 · Augmentation selection** 🟠 `dinov2/train/train.py`
- **What**: pick `MaskingAugmentation3D` when `fmri_augmentation` is set (else the
  official DINO augmentation).
- **Where**: `do_train` augmentation branch (~L294) · import L20.

**7.3 · Loss scaling for gradient accumulation** 🟠 `dinov2/train/ssl_meta_arch.py`
- **What**: divide the accumulated loss by `loss_scale` before `backward`, and guard a
  CUDA `_streams` access.
- **Why**: micro-batch 2 × grad-accum 8 = effective 16 needs correct loss averaging.
- **Where**: `loss_scale` L133, L347 · `_streams` guard L361.

**7.4 · Effective batch size with grad-accum** 🟠 `dinov2/utils/config.py`
- **What**: include `grad_accum_steps` in the effective batch used for LR scaling.
- **Where**: L24.

**7.5 · The training config** 🔵 `dinov2/configs/train/fmri_vits.yaml`
- **What**: all the fMRI knobs — `fmri_mode`, `fmri_img_size [45,54,45]`, `patch_size 9`,
  `fmri_temporal_size 270`, `fmri_temporal_kernel 10`, `proportional_sampler`,
  `fmri_masking_only`, `freeze_pretrained`, LR/epochs.
- **Why**: one config drives the whole fMRI pipeline; the ablations are one-line overrides.
- **Where**: whole file.
