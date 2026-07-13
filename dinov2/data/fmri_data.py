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
  │ 3. COMPOSE EACH BATCH       ProportionalBatchSampler                     │
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
  │ 8. AUGMENT                  MaskingAugmentation3D                        │
  │    Full-volume views + per-token random masking (for iBOT).             │
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
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset, Sampler

from .fmri_const import (                       # noqa: F401  (re-exported)
    LAB_ROOT, TARGET_TR, TARGET_SHAPE, DEFAULT_T_FIXED, DEFAULT_MANIFEST,
    DEFAULT_SPLIT, CORPUS_DATASETS, HOLDOUT_DATASETS, ABIDE_SITE_TR,
    OASIS_DEFAULT_TR, AOMIC_TR, HCP_TR, ADNI_TR,
)

logger = logging.getLogger("dinov2")


# =====================================================================
# 1. CORPUS — read the manifest, apply holdout, index by dataset.
#    (scan discovery + manifest creation are OFFLINE -> dinov2/data/fmri_offline.py)
# =====================================================================

def _index_by_dataset(entries):
    """Group the scans by dataset so the sampler can build balanced batches.

    `entries` is a flat list; the ProportionalBatchSampler needs to know which indices
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
                          split_map=None, holdout_datasets=(), pretrain_splits=("train",)):
    """Read the corpus manifest and produce the exact list of scans training will see.

    This is the NORMAL load path, run once at the start of every training run. It reads
    corpus_manifest.csv row by row (one row = one scan, with its path, native TR and
    pre-computed length) and keeps a scan only if it passes TWO filters:
      1. LENGTH  — drop scans whose upsampled_T (length after TR harmonization) is below
                   min_upsampled_t, i.e. too short to fill a 270-frame window.
      2. HOLDOUT — for a holdout dataset, drop scans of TEST subjects (split not in
                   pretrain_splits). The encoder never sees them -> no leakage.
    Because upsampled_T is pre-computed (offline), both filters run WITHOUT opening any
    scan file. Output is the same shape as build_corpus_entries.

    Args:
      manifest_path    : path to corpus_manifest.csv.
      datasets         : which cohorts to keep.
      min_upsampled_t  : drop scans whose upsampled_T < this (too short for a window);
                         set to t_fixed (270) at training time.
      split_map        : {dataset: {subject: split}} from _load_split_map (or None to
                         keep every subject, i.e. no holdout).
      holdout_datasets : datasets on which the holdout filter applies.
      pretrain_splits  : which splits are kept for holdout datasets (default: ("train",),
                         so test subjects are excluded -> no leakage).
    Returns:
      `entries` (same shape as build_corpus_entries), e.g.:
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
            if split_map and ds in holdout_datasets:          # filter 2: holdout (no leakage)
                sp = split_map.get(ds, {}).get(row["subject_id"])
                if sp is not None and sp not in pretrain_splits:
                    n_holdout += 1
                    continue
            entries.append({"dataset": ds, "path": row["path"],
                            "subject_id": row["subject_id"], "tr": float(row["tr"])})
    if n_short:
        logger.info(f"manifest: dropped {n_short} scans with upsampled_T < {min_upsampled_t}")
    if n_holdout:
        logger.info(f"holdout: excluded {n_holdout} test scans of {holdout_datasets}")
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


def _native_window(T, tr_native, t_fixed, target_tr=TARGET_TR):
    """Choose which slice of a scan to use — a window of fixed REAL duration.

    Every dataset must contribute the same amount of brain activity, but they have
    different TRs, so the same duration means a different number of frames per dataset.
    We want a window spanning t_fixed * target_tr = 270 * 0.72 = 194.4 s of real time;
    at the scan's native TR that is `win = round(194.4 / tr_native)` frames (e.g. 270
    frames for HCP @ 0.72 s, but only ~65 for ADNI @ 3.0 s). The start is picked
    RANDOMLY, which acts as temporal augmentation (a different segment every epoch).
    If the scan is shorter than the window, take it whole — _temporal_resample then
    stretches it up to t_fixed.

    Args:
      T         : native number of frames of the scan.
      tr_native : the scan's native repetition time (s).
      t_fixed   : target window length in frames after harmonization (270).
      target_tr : common TR after harmonization (0.72 s).
    Returns:
      (start, win): start index and window length, in NATIVE frames. _temporal_resample
      then brings `win` frames to exactly t_fixed (270).
    """
    win = max(1, round(t_fixed * target_tr / tr_native))
    if T >= win:
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


def _finalize(clip, t_fixed, target_shape=TARGET_SHAPE):
    """Turn a raw cropped window into the model-ready tensor.

    Small orchestrator called by _load right after the native window is cropped; it
    bundles the three finishing steps, in order:
      1. add a channel dim if the scan is stored as (n, X, Y, Z) -> (n, 1, X, Y, Z);
      2. resize the spatial grid to (45, 54, 45) if it isn't already (trilinear);
      3. resample time to t_fixed=270 @ 0.72 s (_temporal_resample), then z-score each
         frame (_zscore_per_frame).
    After this, every scan — whatever its dataset, TR or native resolution — has the
    EXACT same shape (t_fixed, 1, 45, 54, 45), so a mixed batch can be stacked together.

    Args:
      clip         : tensor (n, X, Y, Z) or (n, 1, X, Y, Z) — the cropped native window.
      t_fixed      : target number of frames (270).
      target_shape : target spatial size (45, 54, 45).
    Returns:
      tensor (t_fixed, 1, *target_shape), z-scored — ready for the model.
    """
    clip = clip.float()
    if clip.ndim == 4:                                # (n,X,Y,Z) -> add channel
        clip = clip.unsqueeze(1)
    if tuple(clip.shape[-3:]) != tuple(target_shape):
        clip = F.interpolate(clip, size=tuple(target_shape), mode="trilinear", align_corners=False)
    return _zscore_per_frame(_temporal_resample(clip, t_fixed))


# =====================================================================
# 3. DATASET + SAMPLER + AUGMENTATION
# =====================================================================

class MixedFMRIDataset(Dataset):
    """The five fMRI cohorts (HCP/ABIDE/OASIS/AOMIC/ADNI) presented as ONE dataset,
    with the same (transform / target_transform) API as ImageNet so it drops into
    DINOv2's do_train unchanged.

    Each __getitem__ returns one preprocessed window:
        image  : float tensor (T_fixed, 1, 45, 54, 45)   -- z-scored, TR-harmonized
        target : 0                                        -- unused (self-supervised)

    `dataset_indices` ({name -> [global indices]}) is exposed so
    ProportionalBatchSampler can compose each batch with a per-dataset quota.

    __init__ does three things (each a small helper below):
        1. _apply_exclude : optionally drop whole datasets from pretraining
        2. _discover      : build the scan list (from the manifest) + holdout
        3. store transforms
    """

    def __init__(self, root=None, *, t_fixed=DEFAULT_T_FIXED, temporal_crop=None,
                 datasets=CORPUS_DATASETS, exclude=None, manifest=None, drop_short=True,
                 split_file=None, holdout_datasets=HOLDOUT_DATASETS,
                 pretrain_splits=("train",), transform=None, target_transform=None,
                 **_ignored):
        # temporal_crop is a legacy alias for t_fixed.
        self.t_fixed = int(temporal_crop if temporal_crop is not None else t_fixed)
        self.transform, self.target_transform = transform, target_transform
        lab_root = root or LAB_ROOT

        datasets = self._apply_exclude(datasets, exclude)
        self.entries, self.dataset_indices = self._discover(
            lab_root, datasets, drop_short, manifest, split_file,
            holdout_datasets, pretrain_splits)
        if not self.entries:
            raise FileNotFoundError(f"No scans under {lab_root} for datasets={datasets}")

    @staticmethod
    def _apply_exclude(datasets, exclude):
        """Drop whole datasets from pretraining, so a downstream probe can later use
        the FULL held-out cohort, encoder unseen.

        Args:
          datasets : the current tuple of dataset names.
          exclude  : names to drop, e.g. "ADNI" or "ADNI,ABIDE" (None -> keep all).
        Returns:
          `datasets` with the excluded names removed.
        """
        if not exclude:
            return datasets
        ex = {d.strip() for d in str(exclude).replace("-", ",").split(",")}
        logger.info(f"MixedFMRIDataset: excluding whole datasets {ex}")
        return tuple(d for d in datasets if d not in ex)

    def _discover(self, lab_root, datasets, drop_short, manifest, split_file,
                  holdout_datasets, pretrain_splits):
        """Build the scan list + its per-dataset index by reading the corpus manifest
        (fast; it carries T_native so short scans AND holdout subjects are filtered
        without opening the files). The manifest is REQUIRED — build it offline first
        with dinov2.data.fmri_offline.write_corpus_manifest.

        Args:
          lab_root         : data root.
          datasets         : cohorts to include.
          drop_short       : if True, drop scans shorter than t_fixed after harmonization.
          manifest         : manifest path (None -> <lab_root>/corpus_manifest.csv).
          split_file       : split path (None -> <lab_root>/subject_split.json).
          holdout_datasets : datasets on which the holdout filter applies.
          pretrain_splits  : splits kept for holdout datasets (("train",)).
        Returns:
          (entries, dataset_indices) — the scan list and {dataset: [indices]}.
        Raises:
          FileNotFoundError if the manifest does not exist.
        """
        # Subject-level holdout via the split file (absent -> keep every subject).
        sf = Path(split_file) if split_file else Path(lab_root) / DEFAULT_SPLIT
        split_map = _load_split_map(sf) if sf.exists() else None

        man = Path(manifest) if manifest else Path(lab_root) / DEFAULT_MANIFEST
        if not man.exists():
            raise FileNotFoundError(
                f"No corpus manifest at {man}. Build it offline first with "
                "dinov2.data.fmri_offline.write_corpus_manifest.")
        entries = entries_from_manifest(
            man, datasets, min_upsampled_t=(self.t_fixed if drop_short else 0),
            split_map=split_map,
            holdout_datasets=holdout_datasets if split_map else (),
            pretrain_splits=pretrain_splits)
        idx = _index_by_dataset(entries)                   # per-dataset indices for the sampler
        counts = {k: len(v) for k, v in idx.items()}
        logger.info(f"MixedFMRIDataset: {len(entries)} scans  T_fixed={self.t_fixed}  "
                    f"target_TR={TARGET_TR}s  manifest={man.name}  per-dataset={counts}")
        return entries, idx

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        """Load one scan and preprocess it into a model-ready window.

        Args:
          idx : global index into self.entries.
        Returns:
          tensor (T_fixed, 1, 45, 54, 45). The native window is cropped FIRST (on the
          mmap) so a 500 MB HCP scan never fully loads; resize + resample-to-0.72s +
          z-score then happen in _finalize.
        """
        e = self.entries[idx]
        scan = _load_mmap(e["path"])                       # lazy (T, X, Y, Z)
        start, win = _native_window(scan.shape[0], e["tr"], self.t_fixed)
        return _finalize(scan[start:start + win].clone(), self.t_fixed)

    def __getitem__(self, idx):
        """Returns (image, target): image = preprocessed + augmented window
        (T_fixed, 1, 45, 54, 45); target = 0 (unused — self-supervised)."""
        image = self._load(idx)
        target: Any = 0                                    # unused (self-supervised)
        if self.transform is not None:
            image = self.transform(image)                  # augmentation (masking, Phase 5)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ProportionalBatchSampler(Sampler):
    """Batches with a fixed per-dataset composition (default quota, batch 16:
    HCP 4, ABIDE 4, OASIS 4, ADNI 3, AOMIC 1). Each dataset's indices are shuffled
    and consumed without replacement; a dataset that runs out mid-epoch is
    reshuffled and cycled (smaller datasets cycle more often). Use with
    DataLoader(dataset, batch_sampler=sampler); pass rank/world_size for DDP."""

    DEFAULT_QUOTA = {"HCP": 4, "ABIDE": 4, "OASIS": 4, "ADNI": 3, "AOMIC": 1}

    def __init__(self, dataset_indices, quota=None, *, batches_per_epoch=None,
                 seed=0, rank=0, world_size=1):
        """Args:
          dataset_indices        : {dataset: [global indices]} (from MixedFMRIDataset).
          quota                  : per-batch count per dataset (default DEFAULT_QUOTA).
          batches_per_epoch      : override; else ceil(total scans / batch_size).
          seed, rank, world_size : DDP — each rank draws a disjoint stream of batches.
        """
        self.dataset_indices = {k: list(v) for k, v in dataset_indices.items() if v}
        self.quota = {k: q for k, q in (quota or self.DEFAULT_QUOTA).items()
                      if k in self.dataset_indices and q > 0}
        if not self.quota:
            raise ValueError(f"No dataset matches quota. Have {list(self.dataset_indices)}, "
                             f"quota {list(quota or self.DEFAULT_QUOTA)}")
        self.batch_size = sum(self.quota.values())
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, max(1, world_size), 0
        if batches_per_epoch is None:
            total = sum(len(self.dataset_indices[k]) for k in self.quota)
            batches_per_epoch = math.ceil(total / self.batch_size)
        self.batches_per_epoch = max(1, batches_per_epoch // self.world_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        """Yield one batch at a time: a list of scan indices with the per-dataset quota
        (e.g. 4 HCP + 4 ABIDE + 4 OASIS + 3 ADNI + 1 AOMIC). Each dataset's pool is
        shuffled; a pool that runs out mid-epoch is reshuffled and cycled (small
        datasets cycle more often); the final batch order is shuffled too."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000 + self.rank)
        pools, ptr = {}, {}
        for name in self.quota:
            idxs = self.dataset_indices[name]
            pools[name] = [idxs[i] for i in torch.randperm(len(idxs), generator=g).tolist()]
            ptr[name] = 0
        for _ in range(self.batches_per_epoch):
            batch = []
            for name, q in self.quota.items():
                pool, p = pools[name], ptr[name]
                for _ in range(q):
                    if p >= len(pool):                     # epoch-local reshuffle + cycle
                        pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
                        pools[name], p = pool, 0
                    batch.append(pool[p])
                    p += 1
                ptr[name] = p
            yield [batch[i] for i in torch.randperm(len(batch), generator=g).tolist()]


class MaskingAugmentation3D:
    """fMRI augmentation = MASKING ONLY (no spatial/temporal crop). All crops are
    the FULL volume; the per-token random masking in the collate (for iBOT) is the
    only corruption. Matches DataAugmentationDINO's call contract so do_train uses
    it as a drop-in. Scale/size args are accepted for API parity but unused."""

    def __init__(self, global_crops_scale=None, local_crops_scale=None,
                 local_crops_number=3, global_crops_size=None, local_crops_size=None,
                 global_crops_number=2):
        self.global_crops_number = int(global_crops_number)
        self.local_crops_number = int(local_crops_number)
        logger.info(f"fMRI MASKING-ONLY augmentation: global={self.global_crops_number} "
                    f"local={self.local_crops_number} (masking applied in collate)")

    def __call__(self, scan):
        """Build the DINO views. No cropping — every view is the FULL volume repeated
        (the per-token masking in the collate is the only corruption).

        Args:
          scan : one preprocessed window (T_fixed, 1, 45, 54, 45).
        Returns:
          dict with `global_crops` / `global_crops_teacher` (x global_crops_number),
          `local_crops` (x local_crops_number), and empty `offsets` — the contract
          do_train expects from DataAugmentationDINO.
        """
        return {"global_crops":         [scan] * self.global_crops_number,
                "global_crops_teacher": [scan] * self.global_crops_number,
                "local_crops":          [scan] * self.local_crops_number,
                "offsets":              ()}
