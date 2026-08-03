"""Multi-source fMRI training data for the DINOv2 pipeline.

WHERE THE DATA COMES FROM (all on disk, prepared offline):
  - raw scans          <lab>/<DATASET>_data/downsampled/.../*.pt  = (T, X, Y, Z) tensors
  - corpus_manifest.csv  one row per scan: path, native TR, native length  [fmri_offline.py]
  - subject_split.json   the 70/30 train/test split, by subject         [make_subject_split]

WHAT WE WANT: one harmonized, z-scored window per scan, ready for the ViT:
                          (T_FIXED=270, 1, 45, 54, 45)

THE PIPELINE (top = raw on disk, bottom = tensor fed to the model). Each box is one
step: the function it calls, and what that function does.

     inputs on disk:  corpus_manifest.csv  ·  subject_split.json  ·  raw *.pt

  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. BUILD THE SCAN LIST      entries_from_manifest (+ _load_split_map)    │
  │    Read the manifest CSV. Drop scans too short for a 270-window, and drop│
  │    the TEST subjects (holdout -> no leakage).                           │
  │    -> entries = [{dataset, path, subject_id, tr}, ...]   (the scan list) │
  └────────────────────────────────────────────────────────────────────────┘
                                     │
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. INDEX BY DATASET         _index_by_dataset(entries)                   │
  │    Group scan indices per dataset -> {dataset: [indices]} for the sampler│
  └────────────────────────────────────────────────────────────────────────┘
                                     │
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. COMPOSE EACH BATCH       ProportionalInfiniteSampler  [samplers.py]   │
  │    Pick scan indices with a fixed quota HCP4/ABIDE4/OASIS4/ADNI3/AOMIC1. │
  └────────────────────────────────────────────────────────────────────────┘
                                     │  for each chosen index i: _load(i)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. OPEN THE SCAN            _load_mmap(path)   -> (T, X, Y, Z), lazy mmap │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 5. CROP A WINDOW            _native_window()   random 270-window ~194.4 s │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 6. HARMONIZE TR             _temporal_resample()  native TR -> 270 @ 0.72s│
  ├────────────────────────────────────────────────────────────────────────┤
  │ 7. NORMALIZE                _zscore_per_frame()   per-frame spatial z-score│
  └────────────────────────────────────────────────────────────────────────┘
                                     │  = (270, 1, 45, 54, 45)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 8. VIEWS                    FullVolumeViews3D                            │
  │    Builds 2 global + N local views (all the full volume). The only aug, │
  │    per-token masking (for iBOT), is applied later in collate.           │
  └────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                    DINOv2 student / teacher  (via do_train)

Sections below: 1. corpus · 2. windowing · 3. dataset + sampler + augmentation.
Offline tools (manifest / T_fixed) live in fmri_offline.py; constants in fmri_const.py.
"""

import csv
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .fmri_const import (                       # noqa: F401  (re-exported)
    LAB_ROOT, TARGET_TR, TARGET_SHAPE, DEFAULT_T_FIXED, DEFAULT_MANIFEST,
    DEFAULT_SPLIT, CORPUS_DATASETS, HOLDOUT_DATASETS, PRETRAIN_SPLITS, DROP_SHORT,
    GLOBAL_CROPS_NUMBER,
)

logger = logging.getLogger("dinov2")


# =====================================================================
# 1. CORPUS — read the manifest, apply holdout, index by dataset.
#    (scan discovery + manifest creation are OFFLINE -> dinov2/data/fmri_offline.py)
# =====================================================================

def _index_by_dataset(entries):
    """Group the scans by dataset so the sampler can build balanced batches.

    `entries` is a flat list; the ProportionalInfiniteSampler needs to know which indices
    belong to which cohort to draw its per-dataset quota (4 HCP, 4 ABIDE, ...). This
    returns {dataset: [indices into entries]}. It is DERIVED from entries in one pass,
    so the loading functions don't have to carry it around — the dataset builds it once.

    Args:
      entries : the scan list from entries_from_manifest.
    Returns:
      {dataset name: [indices into entries]}.

    Example:
      [{"dataset": "HCP", ...}, {"dataset": "ABIDE", ...}, {"dataset": "HCP", ...}]
      -> {"HCP": [0, 2], "ABIDE": [1]}
    """
    by = {}
    for i, e in enumerate(entries):
        by.setdefault(e["dataset"], []).append(i)
    return by


def _load_split_map(split_file):
    """Read subject_split.json and turn it into a fast per-subject lookup.

    The split file lists, per dataset, which SUBJECTS are train vs test. We invert it
    into {dataset: {subject_id: split}} so entries_from_manifest can check any subject's
    membership in O(1) while filtering (for the holdout). Splitting by SUBJECT (not by
    scan) is what prevents leakage: all scans of a held-out subject stay out together.

    Args:
      split_file : path to subject_split.json.
    Returns:
      {dataset: {subject_id: "train"|"test"}}.

    Example:
      file:   {"datasets": {"ADNI": {"train": ["s1", "s2"], "test": ["s3"]}}}
      returns {"ADNI": {"s1": "train", "s2": "train", "s3": "test"}}
    """
    d = json.loads(Path(split_file).read_text())
    return {ds: {s: name for name, subs in splits.items() for s in subs}
            for ds, splits in d.get("datasets", {}).items()}


def entries_from_manifest(manifest_path, datasets=CORPUS_DATASETS, min_upsampled_t=0,
                          split_map=None):
    """Read the corpus manifest and produce the exact list of scans training will see.

    This is the NORMAL load path, run once at the start of every training run. It reads
    corpus_manifest.csv row by row (one row = one scan, with its path, native TR and
    pre-computed length) and keeps a scan only if it passes TWO filters:
      1. LENGTH  — drop scans whose upsampled_T (length after TR harmonization) is below
                   min_upsampled_t, i.e. too short to fill a 270-frame window.
      2. HOLDOUT — for a dataset in HOLDOUT_DATASETS, drop scans of TEST subjects (split
                   not in PRETRAIN_SPLITS). The encoder never sees them -> no leakage.
    HOLDOUT_DATASETS and PRETRAIN_SPLITS are constants (fmri_const), not arguments.
    Because upsampled_T is pre-computed (offline), both filters run WITHOUT opening any
    scan file.

    Args:
      manifest_path   : path to corpus_manifest.csv.
      datasets        : which cohorts to keep.
      min_upsampled_t : drop scans whose upsampled_T < this (too short for a window);
                        set to t_fixed (270) at training time.
      split_map       : {dataset: {subject: split}} from _load_split_map (or None to
                        keep every subject, i.e. no holdout).
    Returns:
      `entries`, e.g.:
      [{"dataset": "HCP", "path": "...", "subject_id": "subject_100206", "tr": 0.72}, ...]
    """
    entries: list = []
    n_short = n_holdout = 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in datasets:
                continue
            if int(row["upsampled_T"]) < min_upsampled_t:     # filter 1: too short
                n_short += 1
                continue
            if split_map and ds in HOLDOUT_DATASETS:          # filter 2: holdout (no leakage)
                sp = split_map.get(ds, {}).get(row["subject_id"])
                if sp is not None and sp not in PRETRAIN_SPLITS:
                    n_holdout += 1
                    continue
            entries.append({"dataset": ds, "path": row["path"],
                            "subject_id": row["subject_id"], "tr": float(row["tr"])})
    if n_short:
        logger.info(f"manifest: dropped {n_short} scans with upsampled_T < {min_upsampled_t}")
    if n_holdout:
        logger.info(f"holdout: excluded {n_holdout} test scans of {HOLDOUT_DATASETS}")
    return entries


# =====================================================================
# 2. WINDOWING — mmap -> native window -> resample to T_FIXED @ 0.72s
# =====================================================================

def _zscore_per_frame(scan):
    """Per-frame spatial z-score: normalize each timepoint volume across its voxels.

    Args:
      scan : tensor (T, 1, X, Y, Z).
    Returns:
      tensor (T, 1, X, Y, Z), each frame zero-mean / unit-std over (1, X, Y, Z)
      (a constant frame maps to zeros).
    """
    mean = scan.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = scan.std(dim=(1, 2, 3, 4), keepdim=True)
    return torch.where(std > 1e-6, (scan - mean) / std.clamp_min(1e-6), torch.zeros_like(scan))


def _load_mmap(path):
    """Memory-map a scan lazily (no RAM used until it is sliced).

    Args:
      path : path to a .pt scan tensor.
    Returns:
      tensor (T, X, Y, Z) or (T, 1, X, Y, Z), memory-mapped on CPU.
    """
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _native_window(T, tr_native, t_fixed):
    """Choose which slice of a scan to use — a window of fixed REAL duration.

    Every dataset must contribute the same amount of brain activity, but they have
    different TRs, so the same duration means a different number of frames per dataset.
    We want a window spanning t_fixed * TARGET_TR = 270 * 0.72 = 194.4 s of real time;
    at the scan's native TR that is `win = round(194.4 / tr_native)` frames (e.g. 270
    frames for HCP @ 0.72 s, but only ~65 for ADNI @ 3.0 s). The start is picked
    RANDOMLY, which acts as temporal augmentation (a different segment every epoch).
    If the scan is shorter than the window, take it whole — _temporal_resample then
    stretches it up to t_fixed. (TARGET_TR is a constant from fmri_const.)

    Args:
      T         : native number of frames of the scan.
      tr_native : the scan's native repetition time (s).
      t_fixed   : target window length in frames after harmonization (270).
    Returns:
      (start, win): start index and window length, in NATIVE frames. _temporal_resample
      then brings `win` frames to exactly t_fixed (270).
    """
    win = max(1, round(t_fixed * TARGET_TR / tr_native))
    if T >= win:
        # DEBUG: FMRI_FIXED_WINDOW pins the start to 0 so every load of a scan returns
        # the EXACT same window (no temporal augmentation) — used to feed a strictly
        # identical input in the overfit sanity checks.
        if os.environ.get("FMRI_FIXED_WINDOW"):
            return 0, win
        return int(np.random.randint(0, T - win + 1)), win
    return 0, T


def _temporal_resample(clip, n_out):
    """Change the number of time frames from n_in to n_out — the actual TR harmonization
    that puts every dataset on the common 0.72 s sampling rate.

    The window has n_in native frames covering 194.4 s; we need exactly n_out = 270
    frames at 0.72 s, so we resample by the rational factor n_out/n_in (e.g. ADNI
    65 -> 270, upsampling ~4.15x). We use scipy.signal.resample_poly (POLYPHASE,
    anti-aliased FIR) rather than linear interpolation: BOLD is a band-limited signal,
    and polyphase avoids the aliasing (spectral folding) that naive interpolation would
    introduce (per Ariel). resample_poly can return +/-1 frame from rounding, so we trim
    or pad the last frame to land on exactly n_out.

    Args:
      clip  : tensor (n_in, 1, X, Y, Z) — the cropped native window.
      n_out : target number of frames (t_fixed = 270).
    Returns:
      tensor (n_out, 1, X, Y, Z). HCP (n_in == n_out, already 0.72 s) is a no-op.
    """
    n_in = clip.shape[0]
    if n_in == n_out:
        return clip
    g = math.gcd(n_out, n_in)
    out = resample_poly(clip.contiguous().numpy(), n_out // g, n_in // g, axis=0)
    out = torch.from_numpy(np.ascontiguousarray(out)).float()
    if out.shape[0] > n_out:                          # guard off-by-one from ceil
        out = out[:n_out]
    elif out.shape[0] < n_out:
        pad = out[-1:].expand(n_out - out.shape[0], *out.shape[1:])
        out = torch.cat([out, pad], dim=0)
    return out.contiguous()


def _finalize(clip, t_fixed):
    """Turn a raw cropped window into the model-ready tensor.

    Small orchestrator called by _load right after the native window is cropped; it
    bundles the three finishing steps, in order:
      1. add a channel dim if the scan is stored as (n, X, Y, Z) -> (n, 1, X, Y, Z);
      2. resize the spatial grid to TARGET_SHAPE=(45, 54, 45) if it isn't already (trilinear);
      3. resample time to t_fixed=270 @ 0.72 s (_temporal_resample), then z-score each
         frame (_zscore_per_frame).
    After this, every scan — whatever its dataset, TR or native resolution — has the
    EXACT same shape (t_fixed, 1, 45, 54, 45), so a mixed batch can be stacked together.

    Args:
      clip    : tensor (n, X, Y, Z) or (n, 1, X, Y, Z) — the cropped native window.
      t_fixed : target number of frames (270).
    Returns:
      tensor (t_fixed, 1, *TARGET_SHAPE), z-scored — ready for the model.
    """
    clip = clip.float()
    if clip.ndim == 4:                                # (n,X,Y,Z) -> add channel
        clip = clip.unsqueeze(1)
    if tuple(clip.shape[-3:]) != tuple(TARGET_SHAPE):
        clip = F.interpolate(clip, size=tuple(TARGET_SHAPE), mode="trilinear", align_corners=False)
    return _zscore_per_frame(_temporal_resample(clip, t_fixed))


# =====================================================================
# 3. DATASET + SAMPLER + AUGMENTATION
# =====================================================================

class MixedFMRIDataset(Dataset):
    """The five fMRI cohorts (HCP/ABIDE/OASIS/AOMIC/ADNI) presented as ONE dataset,
    with the same (root / transform) API as ImageNet so it drops into DINOv2's
    do_train unchanged. SSL-only: __getitem__ returns an empty target, no labels.

    Each __getitem__ returns one preprocessed window:
        image  : float tensor (T_fixed, 1, 45, 54, 45)   -- z-scored, TR-harmonized
        target : 0                                        -- unused (self-supervised)

    `dataset_indices` ({name -> [global indices]}) is exposed so
    ProportionalInfiniteSampler can compose each batch with a per-dataset quota.

    __init__ is minimal: read the manifest into the scan list (_discover) and store
    the transform. Everything else (which datasets, the holdout, the filters) is a
    constant in fmri_const.py.
    """

    def __init__(self, root=None, *, t_fixed=DEFAULT_T_FIXED, transform=None, **_ignored):
        # Only what actually varies is an argument (t_fixed from the config string,
        # transform from do_train). Everything else is a constant in fmri_const.py.
        # target_transform is swallowed by **_ignored: we are SSL-only, so there is no
        # label to transform (see __getitem__).
        self.t_fixed = int(t_fixed)                          # window length in frames (270)
        self.transform = transform                           # image augmentation, or None
        lab_root = root or LAB_ROOT

        self.entries, self.dataset_indices = self._discover(lab_root)
        if not self.entries:
            raise FileNotFoundError(f"No scans under {lab_root} for {CORPUS_DATASETS}")

    def _discover(self, lab_root):
        """Turn the on-disk corpus into this dataset's scan list — the one-time setup
        run by __init__, before any sample is read. Three steps:

          1. Load the subject split (subject_split.json): {subject_id -> "train"/"test"}.
             Its test subjects are the holdout kept OUT of pretraining. File absent -> keep all.
          2. Read the corpus manifest (corpus_manifest.csv, one row per scan). Because each
             row already carries the scan's native length, entries_from_manifest can drop
             short scans AND holdout subjects here WITHOUT opening a single .pt file. The
             manifest is REQUIRED (built offline by fmri_offline.write_corpus_manifest); this
             never globs the disk itself.
          3. Index the surviving scans by dataset ({dataset -> [row positions]}) so the
             ProportionalInfiniteSampler can draw its per-dataset quota.

        What is kept vs dropped is governed entirely by constants in fmri_const
        (HOLDOUT_DATASETS, PRETRAIN_SPLITS, DROP_SHORT), never by arguments.

        Args:
          lab_root : data root holding corpus_manifest.csv and subject_split.json.
        Returns:
          (entries, dataset_indices) — the flat scan list and {dataset: [indices]}.
        Raises:
          FileNotFoundError : the manifest is missing (build it offline first).
        """
        # Step 1 — subject-level holdout map (absent file -> keep every subject).
        sf = Path(lab_root) / DEFAULT_SPLIT
        split_map = _load_split_map(sf) if sf.exists() else None

        # Step 2 — read the manifest; drop short scans + holdout subjects from its rows.
        man = Path(lab_root) / DEFAULT_MANIFEST
        if not man.exists():
            raise FileNotFoundError(
                f"No corpus manifest at {man}. Build it offline first with "
                "dinov2.data.fmri_offline.write_corpus_manifest.")
        entries = entries_from_manifest(
            man, min_upsampled_t=(self.t_fixed if DROP_SHORT else 0), split_map=split_map)

        # DEBUG overfit sanity check: FMRI_OVERFIT_N=k keeps only k HCP scans, so the
        # model is asked to MEMORIZE a handful of samples. A healthy model must drive the
        # loss toward 0 here; if it does not, learning is mechanically broken (bad gradient
        # flow / frozen weights / degenerate target), NOT a data problem. We fix the cohort
        # to HCP (clean, fully preprocessed reference) so the run is reproducible and the
        # sampler only needs one cohort.
        overfit_n = int(os.environ.get("FMRI_OVERFIT_N", "0"))
        if overfit_n > 0:
            entries = [e for e in entries if e["dataset"] == "HCP"][:overfit_n]
            if not entries:
                raise RuntimeError("FMRI_OVERFIT_N set but no HCP scan found in the manifest.")
            logger.warning(f"FMRI_OVERFIT_N={overfit_n}: corpus truncated to {len(entries)} "
                           f"HCP scan(s) for an overfit sanity check — NOT a normal run.")
            # Print exactly which scans the model will memorize (dataset, subject, path).
            for i, e in enumerate(entries):
                logger.warning(f"  overfit scan[{i}]  dataset={e['dataset']:6s} "
                               f"subject={e['subject_id']}  tr={e['tr']}  path={e['path']}")

        # Step 3 — group the kept scans by dataset for the proportional sampler.
        idx = _index_by_dataset(entries)
        counts = {k: len(v) for k, v in idx.items()}
        logger.info(f"MixedFMRIDataset: {len(entries)} scans  T_fixed={self.t_fixed}  "
                    f"target_TR={TARGET_TR}s  manifest={man.name}  per-dataset={counts}")
        return entries, idx

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        """One scan -> one model-ready window (T_fixed, 1, 45, 54, 45). Three gestures:
          1. mmap the file          — lazy, nothing loaded yet.
          2. pick a native window   — random start, native length covering 194.4s.
          3. crop it, then _finalize — resize + resample to 0.72s + z-score.
        Cropping on the mmap BEFORE .clone() is what keeps a 500 MB HCP scan off RAM:
        only the ~270-frame window is ever materialized.

        Args:
          idx : global index into self.entries.
        Returns:
          tensor (T_fixed, 1, 45, 54, 45), ready for the augmentation transform.
        """
        e = self.entries[idx]
        scan = _load_mmap(e["path"])                       # 1. lazy (T, X, Y, Z)
        start, win = _native_window(scan.shape[0], e["tr"], self.t_fixed)  # 2. window
        return _finalize(scan[start:start + win].clone(), self.t_fixed)    # 3. crop + prep

    def __getitem__(self, idx):
        """Returns (image, target). image = preprocessed window (T_fixed, 1, 45, 54, 45),
        augmented if a transform is set. target is always the empty tuple (): we are
        SSL-only, there is no label — () is the collate-safe "no label" placeholder the
        DINO collator expects (it reads s[0] only, never the target)."""
        image = self._load(idx)
        if self.transform is not None:
            image = self.transform(image)                  # augmentation (masking, Phase 5)
        return image, ()                                   # () instead of labels because we work on SSL


# NOTE: the per-batch quota sampler is ProportionalInfiniteSampler (dinov2/data/samplers.py),
# built via SamplerType.PROPORTIONAL in loaders.py. It is the INFINITE (iteration-based)
# variant DINOv2's training loop needs; MixedFMRIDataset only has to expose dataset_indices.


# Two augmentation knobs (env, read once at import) — for the "MNI-registered brain"
# experiments (Ariel/meeting 2026-08). The registered, z-scored volume must NOT get
# flip or spatial zoom (those break anatomical correspondence); the only image-space
# corruption we allow is additive Gaussian noise.
#   FMRI_VIEW_NOISE_STD       : sigma of independent per-view Gaussian noise (0 = off).
#   FMRI_LOCAL_TEMPORAL_FRAMES: if >0, each LOCAL crop is a random temporal sub-window
#                               of this many frames (a "temporal zoom"); globals stay
#                               the full T_fixed volume. 0 = locals are full volume too.
_VIEW_NOISE_STD = float(os.environ.get("FMRI_VIEW_NOISE_STD", "0.0"))
_LOCAL_TEMPORAL_FRAMES = int(os.environ.get("FMRI_LOCAL_TEMPORAL_FRAMES", "0"))


class FullVolumeViews3D:
    """Produces the DINO view dict (2 global + N local) for MNI-registered fMRI.

    A brain volume is registered to MNI and z-scored, so DINO's usual view
    augmentations (spatial RandomResizedCrop, flip, blur, solarize, colour) would
    break anatomical correspondence and are dropped on purpose (meeting 2026-06-14,
    §2). The ONLY image-space corruption we allow is additive Gaussian noise —
    `FMRI_VIEW_NOISE_STD` sets its sigma, applied INDEPENDENTLY to every view so the
    two globals (and the locals) genuinely differ. Per-token random masking (iBOT)
    is still applied later, per batch, in collate_data_and_cast.

    Two regimes (env-selected):
      - Test 1 (FMRI_LOCAL_TEMPORAL_FRAMES=0): locals have the SAME dimensions as the
        globals — the full T_fixed volume — plus their own noise. No zoom at all.
      - Test 2 (FMRI_LOCAL_TEMPORAL_FRAMES=L>0): each local is a random L-frame
        temporal sub-window ("temporal zoom") + noise; globals stay full-volume.
        Requires the T_eff-aware positional embedding (PositionEmbedding3D.
        combined_patch_pos slices to the crop's temporal length).

    Crop counts: global = GLOBAL_CROPS_NUMBER (fmri_const, the DINO invariant, 2);
    local = local_crops_number (must equal cfg.crops.local_crops_number, which the
    DINO loss reads — single source of truth). **_ignored absorbs DataAugmentationDINO's
    dead scale/size args if a caller still passes them.
    """

    def __init__(self, local_crops_number, **_ignored):
        self.global_crops_number = GLOBAL_CROPS_NUMBER
        self.local_crops_number = int(local_crops_number)
        self.noise_std = _VIEW_NOISE_STD
        self.local_temporal_frames = _LOCAL_TEMPORAL_FRAMES
        logger.info(
            f"fMRI views: global={self.global_crops_number} local={self.local_crops_number} "
            f"noise_std={self.noise_std} local_temporal_frames={self.local_temporal_frames} "
            f"(masking still applied later in collate)"
        )

    def _noise(self, v):
        """Independent additive Gaussian noise (no-op if noise_std == 0)."""
        if self.noise_std > 0:
            return v + torch.randn_like(v, dtype=torch.float32).to(v.dtype) * self.noise_std
        return v

    def _local_view(self, scan):
        """One local view: optional random temporal sub-window, then its own noise."""
        L = self.local_temporal_frames
        if L > 0 and scan.shape[0] > L:
            start = int(torch.randint(0, scan.shape[0] - L + 1, (1,)).item())
            v = scan[start:start + L]
        else:
            v = scan
        return self._noise(v)

    def __call__(self, scan):
        """Build the DINO views.

        Args:
          scan : one preprocessed window (T_fixed, 1, 45, 54, 45).
        Returns:
          dict with `global_crops` / `global_crops_teacher` (x global_crops_number),
          `local_crops` (x local_crops_number), and empty `offsets` — the contract
          do_train expects from DataAugmentationDINO. Globals share the same noised
          tensors for student & teacher (official DINOv2: same input, EMA-different nets).
        """
        globals_ = [self._noise(scan) for _ in range(self.global_crops_number)]
        locals_ = [self._local_view(scan) for _ in range(self.local_crops_number)]
        return {"global_crops":         globals_,
                "global_crops_teacher": globals_,
                "local_crops":          locals_,
                "offsets":              ()}
