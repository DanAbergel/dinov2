"""Multi-source fMRI training data for the DINOv2 pipeline.

Three pieces, in order:
  1. corpus       — discover scans (glob or manifest), apply subject holdout
  2. windowing    — mmap a scan, crop a native window, resample to T_FIXED @ 0.72s
  3. MixedFMRIDataset + ProportionalBatchSampler + MaskingAugmentation3D

MixedFMRIDataset yields ONE harmonized window per scan as a z-scored
(T_FIXED, 1, 45, 54, 45) tensor; ProportionalBatchSampler composes each batch
with a fixed per-dataset quota; MaskingAugmentation3D is the (masking-only)
augmentation. All constants live in fmri_const.py (re-exported here for the
probe and the data-prep tasks that import them from this module).
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
# 1. CORPUS — discover scans + subject-level holdout
# =====================================================================

# ---------------------------------------------------------------------------
# Where each source lives on disk, and how to read its subject id + native TR.
# One declarative entry per dataset -> build_corpus_entries just loops over this,
# so adding a cohort = adding one line here (no new if/elif branch).
#   glob    : path pattern (under lab_root) matching every scan .pt of the cohort.
#   subject : derive the subject id from a scan's Path. All scans of one subject
#             MUST share the same id, so they never straddle the train/test split.
#   tr      : the native repetition time (s). ABIDE differs per acquisition site,
#             so its TR is looked up from the filename -> a function, not a constant.
# ---------------------------------------------------------------------------
DATASET_SOURCES = {
    # HCP: one rest run per subject; subject id = the "subject_XXXXXX" folder name.
    "HCP":   dict(glob="HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt",
                  subject=lambda p: p.parent.name,
                  tr=lambda p: HCP_TR),
    # ABIDE: multi-site; the site is the filename prefix and the TR is per site.
    "ABIDE": dict(glob="ABIDE_data/downsampled/**/*.pt",
                  subject=lambda p: p.stem,
                  tr=lambda p: ABIDE_SITE_TR.get(p.name.split("_")[0])),
    # OASIS-3: one file per session folder; a single documented TR.
    "OASIS": dict(glob="OASIS3_data/downsampled/*/rest_*.pt",
                  subject=lambda p: p.parent.name,
                  tr=lambda p: OASIS_DEFAULT_TR),
    # AOMIC: two protocols (piop1/piop2) with different TRs, told apart by the path.
    "AOMIC": dict(glob="AOMIC_data/downsampled/*/sub-*/restingstate_downsampled.pt",
                  subject=lambda p: p.parent.name,
                  tr=lambda p: AOMIC_TR["piop1" if "piop1" in str(p).lower() else "piop2"]),
    # ADNI: files named I<image_id>.pt inside a per-subject folder.
    "ADNI":  dict(glob="ADNI_data/downsampled/*/I*.pt",
                  subject=lambda p: p.parent.name,
                  tr=lambda p: ADNI_TR),
}


def build_corpus_entries(lab_root=LAB_ROOT, datasets=CORPUS_DATASETS):
    """Discover every scan of the requested datasets on disk.

    Returns (entries, by_dataset):
      entries    : flat list of {dataset, path, subject_id, tr}, one dict per scan.
      by_dataset : {name -> [indices into entries]} — feeds ProportionalBatchSampler
                   so each batch can be composed with a fixed per-dataset quota.

    Example:
      entries = [
        {"dataset": "HCP",   "path": ".../subject_100206/...pt", "subject_id": "subject_100206", "tr": 0.72},
        {"dataset": "HCP",   "path": ".../subject_100307/...pt", "subject_id": "subject_100307", "tr": 0.72},
        {"dataset": "ABIDE", "path": ".../NYU_0051091.pt",       "subject_id": "NYU_0051091",    "tr": 2.0 },
        ...
      ]
      by_dataset = {"HCP": [0, 1, ...], "ABIDE": [2, ...], ...}   # indices into `entries`

    The per-dataset details live in DATASET_SOURCES above; this loop is generic.
    """
    entries, by_dataset = [], {}
    for name in datasets:
        src = DATASET_SOURCES.get(name)
        if src is None:                                    # unknown dataset name -> skip
            continue
        # sorted() makes the file order deterministic (reproducible corpus).
        for p in sorted(Path(lab_root).glob(src["glob"])):
            tr = src["tr"](p)
            if tr is None:                                 # e.g. an unmapped ABIDE site
                logger.warning(f"{name}: no native TR for {p.name}; skipping")
                continue
            # record this scan's global index under its dataset (for the sampler)...
            by_dataset.setdefault(name, []).append(len(entries))
            # ...then append the scan itself.
            entries.append({"dataset": name, "path": str(p),
                            "subject_id": src["subject"](p), "tr": float(tr)})
    return entries, by_dataset


def write_corpus_manifest(out_path, lab_root=LAB_ROOT, datasets=CORPUS_DATASETS):
    """Offline: scan every scan's native T once and write the corpus manifest CSV
    (dataset,path,subject_id,tr,T_native,upsampled_T). MixedFMRIDataset then reads
    this instead of re-scanning shapes, using upsampled_T to drop too-short scans."""
    entries, _ = build_corpus_entries(lab_root, datasets)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "path", "subject_id", "tr", "T_native", "upsampled_T"])
        for n, e in enumerate(entries, 1):
            T = int(_load_mmap(e["path"]).shape[0])
            w.writerow([e["dataset"], e["path"], e["subject_id"], e["tr"], T,
                        round(T * e["tr"] / TARGET_TR)])
            if n % 200 == 0:
                logger.info(f"  corpus manifest: {n}/{len(entries)}")
    logger.info(f"corpus manifest written: {len(entries)} scans -> {out_path}")
    return out_path


def _load_split_map(split_file):
    """subject_split.json -> {dataset: {subject_id: 'train'|'test'}}."""
    d = json.loads(Path(split_file).read_text())
    return {ds: {s: name for name, subs in splits.items() for s in subs}
            for ds, splits in d.get("datasets", {}).items()}


def entries_from_manifest(manifest_path, datasets=CORPUS_DATASETS, min_upsampled_t=0,
                          split_map=None, holdout_datasets=(), pretrain_splits=("train",)):
    """Build (entries, name->indices) from the corpus manifest CSV. Keeps rows whose
    dataset is requested and whose upsampled_T >= min_upsampled_t. If split_map is
    given, a holdout-dataset scan is kept only if its subject's split is in
    pretrain_splits (test excluded from pretraining -> no leakage)."""
    entries: list = []
    by_dataset: dict = {}
    n_short = n_holdout = 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in datasets:
                continue
            if int(row["upsampled_T"]) < min_upsampled_t:
                n_short += 1
                continue
            if split_map and ds in holdout_datasets:
                sp = split_map.get(ds, {}).get(row["subject_id"])
                if sp is not None and sp not in pretrain_splits:
                    n_holdout += 1
                    continue
            by_dataset.setdefault(ds, []).append(len(entries))
            entries.append({"dataset": ds, "path": row["path"],
                            "subject_id": row["subject_id"], "tr": float(row["tr"])})
    if n_short:
        logger.info(f"manifest: dropped {n_short} scans with upsampled_T < {min_upsampled_t}")
    if n_holdout:
        logger.info(f"holdout: excluded {n_holdout} test scans of {holdout_datasets}")
    return entries, by_dataset


def compute_t_fixed_max(lab_root=LAB_ROOT, datasets=CORPUS_DATASETS, margin=0):
    """Offline: largest T_fixed that fits EVERY scan with no padding = global min
    upsampled length. Returns (t_fixed_max - margin, per_dataset_min, shortest_entry,
    per_dataset_all_upsampled). Use to pick DEFAULT_T_FIXED."""
    entries, _ = build_corpus_entries(lab_root, datasets)
    per, per_all, g_min, argmin = {}, {}, None, None
    for e in entries:
        up = round(_load_mmap(e["path"]).shape[0] * e["tr"] / TARGET_TR)
        d = e["dataset"]
        per[d] = min(per.get(d, up), up)
        per_all.setdefault(d, []).append(up)
        if g_min is None or up < g_min:
            g_min, argmin = up, e
    return max(1, g_min - margin), per, argmin, per_all


# =====================================================================
# 2. WINDOWING — mmap -> native window -> resample to T_FIXED @ 0.72s
# =====================================================================

def _zscore_per_frame(scan):
    """Per-frame z-score on (T, 1, X, Y, Z)."""
    mean = scan.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = scan.std(dim=(1, 2, 3, 4), keepdim=True)
    return torch.where(std > 1e-6, (scan - mean) / std.clamp_min(1e-6), torch.zeros_like(scan))


def _load_mmap(path):
    """Memory-map a scan lazily (no RAM until sliced). Shape (T,X,Y,Z) or (T,1,X,Y,Z)."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _native_window(T, tr_native, t_fixed, target_tr=TARGET_TR):
    """(start, win): a native window of `win` frames spanning t_fixed*target_tr
    seconds, at a random start. Scan shorter than the window -> take it whole
    (it gets stretched up to t_fixed by _temporal_resample)."""
    win = max(1, round(t_fixed * target_tr / tr_native))
    if T >= win:
        return int(np.random.randint(0, T - win + 1)), win
    return 0, T


def _temporal_resample(clip, n_out):
    """(n_in,1,X,Y,Z) -> (n_out,1,X,Y,Z) by polyphase resampling along time
    (scipy.signal.resample_poly — anti-aliased FIR, correct for band-limited BOLD).
    Resamples the window from its native TR to TARGET_TR. HCP (n_in==n_out): no-op."""
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
    """Native window -> (t_fixed, 1, *target_shape), z-scored."""
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
    """The five fMRI cohorts (HCP/ABIDE/OASIS/AOMIC/ADNI) as one dataset. Each
    __getitem__ mmaps a native (T,X,Y,Z) scan, crops a native window spanning
    T_fixed*0.72s (crop FIRST so a 524 MB HCP file never fully loads), resamples
    it to T_fixed frames @ 0.72s, and z-scores -> (T_fixed, 1, 45, 54, 45).
    `dataset_indices` (name -> global indices) feeds ProportionalBatchSampler."""

    def __init__(self, root=None, *, t_fixed=DEFAULT_T_FIXED, temporal_crop=None,
                 datasets=CORPUS_DATASETS, exclude=None, manifest=None, drop_short=True,
                 split_file=None, holdout_datasets=HOLDOUT_DATASETS,
                 pretrain_splits=("train",), transform=None, target_transform=None,
                 **_ignored):
        self.t_fixed = int(temporal_crop if temporal_crop is not None else t_fixed)
        lab_root = root or LAB_ROOT

        # Drop whole datasets from pretraining (e.g. exclude=ADNI for a clean
        # downstream probe over the FULL ADNI cohort, encoder unseen).
        if exclude:
            ex = {d.strip() for d in str(exclude).replace("-", ",").split(",")}
            datasets = tuple(d for d in datasets if d not in ex)
            logger.info(f"MixedFMRIDataset: excluding whole datasets {ex}")

        # Subject-level holdout via split file (absent -> use all data).
        sf = Path(split_file) if split_file else Path(lab_root) / DEFAULT_SPLIT
        split_map = _load_split_map(sf) if sf.exists() else None

        # Prefer the manifest (fast + carries T_native to drop too-short scans).
        man = Path(manifest) if manifest else Path(lab_root) / DEFAULT_MANIFEST
        if man.exists():
            min_t = self.t_fixed if drop_short else 0
            self.entries, self.dataset_indices = entries_from_manifest(
                man, datasets, min_upsampled_t=min_t, split_map=split_map,
                holdout_datasets=holdout_datasets if split_map else (),
                pretrain_splits=pretrain_splits)
            src = f"manifest {man.name}"
        else:
            self.entries, self.dataset_indices = build_corpus_entries(lab_root, datasets)
            src = "glob (no manifest; short scans stretched, not dropped)"

        if not self.entries:
            raise FileNotFoundError(f"No scans under {lab_root} for datasets={datasets}")
        self.transform = transform
        self.target_transform = target_transform
        counts = {k: len(v) for k, v in self.dataset_indices.items()}
        logger.info(f"MixedFMRIDataset: {len(self.entries)} scans  T_fixed={self.t_fixed}  "
                    f"target_TR={TARGET_TR}s  src={src}  per-dataset={counts}")

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        e = self.entries[idx]
        scan = _load_mmap(e["path"])                       # (T,X,Y,Z) lazy
        start, win = _native_window(scan.shape[0], e["tr"], self.t_fixed)
        return _finalize(scan[start:start + win].clone(), self.t_fixed)

    def __getitem__(self, idx):
        image = self._load(idx)
        target: Any = 0
        if self.transform is not None:
            image = self.transform(image)
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
        return {"global_crops":         [scan] * self.global_crops_number,
                "global_crops_teacher": [scan] * self.global_crops_number,
                "local_crops":          [scan] * self.local_crops_number,
                "offsets":              ()}
