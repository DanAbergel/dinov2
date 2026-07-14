# Code review — our fMRI changes on top of DINOv2

Every change we made to the official DINOv2 (`facebookresearch/dinov2`), traced in
the **order the data flows through the pipeline**. Tags (**verified against `upstream/main`**):
🟢 **new file** (entirely ours) · 🟠 **official file we modified** · 🔵 **config**.
Each item = **What** / **Why** (vs official) / **Where** (`file : symbol`).

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
- **Where**: ALL offline, in `dinov2/data/fmri_offline.py`: `DATASET_SOURCES` (declarative
  per-dataset table) · `build_corpus_entries` (glob discovery) · `write_corpus_manifest`
  (writes the CSV). These run ONCE at data-prep, never during training. At training,
  `MixedFMRIDataset._discover` REQUIRES the manifest (raises if missing — no glob fallback).

**1.3 · Subject-level holdout, 70/30 train/test (no leakage)** 🟢 `dinov2/data/fmri_data.py`
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

**1.5 · Per-batch dataset quota (proportional sampling)** 🟠 (new class in modified files)
- **What**: an INFINITE index stream whose every consecutive `batch_size` block has a fixed
  composition — HCP 4 / ABIDE 4 / OASIS 4 / ADNI 3 / AOMIC 1 = 16 — instead of uniform sampling.
- **Why**: the sources differ hugely in size; without a quota HCP dominates every batch and
  AOMIC is almost never seen.
- **How** (reviewed): modelled on the official `InfiniteSampler` — same skeleton (`__iter__` =
  `islice(_iterator(), advance)` for checkpoint resume, infinite `while True`, `seed`/`advance`/
  `rank` args). The ONLY change: it emits quota-blocks (draw `q` from each dataset's shuffled
  pool; a small pool that runs out is reshuffled + cycled) instead of a flat permutation, and it
  does NOT `[start::step]`-stride — each DDP rank seeds its OWN full stream (via `rank`). Contract:
  the loader `batch_size` must divide `sum(quota)` so a micro-batch aligns with a quota block.
- **Cleanup**: dropped the dead `world_size` param (assigned, never read — we don't stride).
- **Where**: `data/samplers.py : ProportionalInfiniteSampler` (new class added to the official
  `samplers.py`) · `data/loaders.py` (edits: `name=="Mixed"`, `SamplerType.PROPORTIONAL`,
  block-size validation). `MixedFMRIDataset` only exposes `dataset_indices`.

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

**3.1 · Spatial resize to (45, 54, 45)** 🟢 `dinov2/data/fmri_data.py`
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
- **Where**: `FullVolumeViews3D` L327 · `__call__` L341.

**5.2 · Per-token random masking (for iBOT)** 🟠 `dinov2/data/masking.py`
- **What**: a NEW class `RandomTokenMaskingGenerator` added to the official `masking.py` —
  MAE-style random token masking over the flattened (T_eff × N_spatial) grid, instead of
  DINOv2's 2D BeiT block masking (`MaskingGenerator`, kept unchanged).
- **Why**: the fMRI token order doesn't respect 2D neighborhoods, so block masking
  doesn't apply.
- **Where**: `RandomTokenMaskingGenerator`.

**5.3 · Mask-generator wiring for fMRI** 🟠 `dinov2/train/train.py`
- **What**: compute `n_tokens` and pick the mask generator from the fMRI token grid
  (T_eff × N_spatial) rather than the official 2D `(img/p, img/p)`.
- **Where**: `do_train` L253 (fmri branch).

---

## Phase 6 — Embedding: the 3D+1D patchify

*Where DINOv2 has one `Conv2d` that cuts a 2D image into patches, we need to cut a 4D
`(T, X, Y, Z)` volume into tokens. This whole file (`patch_embed_3d_plus_1d.py`) is ours.*

**6.1 · `Conv3Plus1d` — the atomic op** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: a **factorised 4D conv** = a 3D spatial conv **then** a 1D temporal conv. PyTorch
  has no `Conv4d` (and a true one would be huge), so we split it — the fMRI analogue of
  MovieGen's `Conv2Plus1d` (video = 2D spatial + 1D temporal).
- **How** (the reshape trick): fold every other axis into the batch so each PyTorch op only
  sees the axis it acts on.
  ```
  (B, C, T, X, Y, Z)
     │  fold T into batch
  (B·T,  C,  X,Y,Z)   ──Conv3d spatial──►  (B·T,  C', X',Y',Z')   ← same 3D filter per frame
     │  fold space into batch
  (B·X'Y'Z',  C',  T) ──Conv1d temporal──► (B·X'Y'Z', C', T')     ← same 1D filter per voxel
     │  unfold
  (B, C', T', X', Y', Z')
  ```
- **Why**: a separable 4D conv — far fewer params than Conv4d, only standard ops. `K/S/P_s`
  tune the spatial axis, `K/S/P_t` the temporal axis, independently.
- **Where**: `Conv3Plus1d`.

**6.2 · Hierarchical patchify** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: stack `Conv3Plus1d` into 3 levels that alternate **process** (`_ResBlock3Plus1d` =
  two Conv3Plus1d + residual) and **reduce** (down + AvgPool), channels growing as resolution
  shrinks (1 → 32 → 64 → 384):
  ```
  input                      (B,  1,  270, 45, 54, 45)
  conv_in + pool /3 spatial  (B, 32,  270, 15, 18, 15)
  block_0  (ResBlock @32)    (B, 32,  270, 15, 18, 15)
  down_0 + pool /3, /2       (B, 64,  135,  5,  6,  5)
  block_1  (ResBlock @64)    (B, 64,  135,  5,  6,  5)
  down_1 + pool /5 temporal  (B, 384,  27,  5,  6,  5)
  block_2  (ResBlock @384)   (B, 384,  27,  5,  6,  5)
     │  rearrange → tokens
  output                     (B, 4050, 384)      = 27 temporal × 150 spatial
  ```
  Spatial /9 total = `patch_size 9` (→ 5×6×5 = 150). Temporal /10 total = `temporal_kernel`
  (→ 27). `down_1_kt = temporal_kernel // 2` is **derived from the config** (not hardcoded) so
  a checkpoint is rebuildable from its saved `config.yaml`.
- **Why**: turn the volume into the 4050-token grid the ViT consumes.
- **Where**: `PatchEmbed3DPlus1D` · `_ResBlock3Plus1d` · `_encoder_forward`.

**6.3 · AvgPool downsampling** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: every conv is **stride-1**; all spatial/temporal reduction is done by **AvgPool**
  (`_spool` / `_tpool`) — a fixed average of a `k` block, **no learned params** (vs a learned
  strided conv). Same fold-into-batch trick (`avg_pool3d` for space, `avg_pool1d` for time).
- **Why**: our design choice — pooling instead of strided conv for the downsampling.
- **Where**: `_spool` · `_tpool`, called in `_encoder_forward` (`spool_k=3`, `pool0_k=2`,
  `pool1_k=down_1_kt`).

**6.4 · Positional embedding (learned, factorised)** 🟢 `dinov2/layers/patch_embed_3d_plus_1d.py`
- **What**: the transformer sees tokens as a set (no order), so we ADD a learned position vector.
  Instead of one flat table `(4050 × 384)`, we keep **two small tables** and broadcast-sum them:
  ```
  pos_temporal (27) ─ repeated over the 150 spatial slots ─┐
                                                            ├─ sum → 4050 positions
  pos_spatial (150) ─ repeated over the 27 time slots ─────┘
  token(t, n) = pos_temporal[t] + pos_spatial[n]     (+ a separate pos_cls for CLS)
  ```
- **Why**: `(27+150+1)×384 = 68 352` params instead of `4050×384 ≈ 1.55 M` — **O(T+N)** not
  **O(T·N)**. The `(t n)` order MUST match the token order `'b (t x y z) c'` from 6.2.
- **Where**: `PositionEmbedding3D` · `combined_patch_pos` (carried by the module, added by the
  ViT in 6.6 — NOT in the patchify forward).

**6.5 · Bind the patchify into the model** 🟠 `dinov2/models/__init__.py`
- **What**: the official ViT already has an `embed_layer=` hook (default = 2D `PatchEmbed`). When
  `cfg.student.fmri_mode` is set we inject `partial(PatchEmbed3DPlus1D, temporal_size,
  temporal_kernel)` as `embed_layer`, and override `img_size` with the 3D `(45,54,45)`.
  ```
  official :  ViT(embed_layer = PatchEmbed [Conv2d, 2D])
  fMRI     :  ViT(embed_layer = partial(PatchEmbed3DPlus1D, temporal_size, temporal_kernel))
  ```
- **Why**: swap the 2D patchify for our 3D+1D one **without forking the ViT or the loop**. The
  ViT calls `embed_layer(img_size, patch_size, in_chans, embed_dim)` — the standard `PatchEmbed`
  contract; `functools.partial` pre-binds the extra fMRI args (`temporal_size`/`temporal_kernel`)
  the ViT doesn't know about, so our richer class fits the standard contract.
- **Where**: `build_model_from_cfg` (fmri branch builds the partial) · `build_model` (forwards
  `embed_layer` to the ViT).

**6.6 · ViT accepts 6D fMRI input** 🟠 `dinov2/models/vision_transformer.py`
- **What**: an early `if x.ndim == 6` branch in `prepare_tokens_with_masks` handles
  `(B, T, C, X, Y, Z)`. Same steps as the official 2D path, but with our factorised pos:
  ```
  patch_embed → (B, 4050, D)
    1. torch.where(masks, mask_token, x)      ← iBOT masking (BEFORE pos, as official)
    2. + combined_patch_pos()                 ← factorised pos (NOT the flat self.pos_embed)
    3. prepend CLS (+ pos_cls)
    4. insert register tokens (no pos)
  → transformer blocks
  ```
- **Why**: the official ViT expects 2D images + a flat `self.pos_embed`; fMRI is 6D and uses the
  factorised pos carried by `PatchEmbed3DPlus1D`. The official 4D path below is untouched;
  `self.pos_embed` is still allocated but unused in fMRI mode (harmless dead weight).
- **Where**: `prepare_tokens_with_masks` (6D branch).

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
- **What**: pick `FullVolumeViews3D` when `fmri_augmentation` is set (else the
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
