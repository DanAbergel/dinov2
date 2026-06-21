# fMRI data utilities — drop-in for DINOv2's official pipeline.
#
# Two Dataset classes (HCP, ADNI). Each returns ONE full scan as a
# (T, 1, X, Y, Z) z-scored tensor. The temporal dimension is kept intact
# and reduced by the PatchEmbed3DPlus1D layer.
#
# One MultiCrop3D class — 3D spatial multi-crop only, no photometric augs.
# Its constructor matches `DataAugmentationDINO` so it drops into
# `do_train` as a one-line replacement (via cfg.train.fmri_augmentation).
# `global_crops_size` / `local_crops_size` are accepted but ignored: fMRI
# crops are resized back to the input volume shape (X, Y, Z) so the ViT
# pos_embed table stays one fixed size.
#
# No colour jitter / blur / solarize / flip / noise / ImageNet normalize —
# fMRI volumes are already z-scored.

import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset, Sampler


logger = logging.getLogger("dinov2")


# ----------------------------------------------------------------------
# Datasets — same `transform=` / `target_transform=` API as ImageNet so
# they plug into `make_dataset` and `do_train` unchanged.
# ----------------------------------------------------------------------

def _zscore_per_frame(scan: torch.Tensor) -> torch.Tensor:
    """Per-frame z-score on (T, 1, X, Y, Z)."""
    mean = scan.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = scan.std(dim=(1, 2, 3, 4), keepdim=True)
    return torch.where(
        std > 1e-6,
        (scan - mean) / std.clamp_min(1e-6),
        torch.zeros_like(scan),
    )


# ----------------------------------------------------------------------
# Multi-source corpus: paths, per-dataset native TR, and the temporal
# harmonization used by MixedFMRIDataset.
#
# Every scan on disk is a (T, X, Y, Z) float32 tensor at a fixed spatial
# resolution of (45, 54, 45). The ONLY thing that differs across datasets is
# the acquisition TR (seconds between consecutive volumes). To mix them we
# resample every scan to a common TR (TARGET_TR = 0.72 s = HCP's native TR,
# chosen to preserve HCP's temporal richness) and crop a fixed-length window of
# T_FIXED frames. See _native_window + _temporal_resample.
# ----------------------------------------------------------------------

LAB_ROOT = "/sci/labs/arieljaffe/dan.abergel1"
TARGET_TR = 0.72                       # common TR after harmonization (HCP native)
TARGET_SHAPE = (45, 54, 45)
# T_fixed = 270 frames @ 0.72s = 194.4s window. Chosen at the knee of the
# window-vs-scans trade-off: it sits 1 frame under ABIDE's min upsampled length
# (271) so ALL of ABIDE is kept, and drops only the 2 short OASIS outliers whose
# upsampled length is < 270 (filtered via the corpus manifest; see build_corpus_
# manifest / MixedFMRIDataset(manifest=...)).
DEFAULT_T_FIXED = 270
DEFAULT_MANIFEST = "corpus_manifest.csv"   # under LAB_ROOT; auto-used if present

# ABIDE I native TR per site (seconds). Site = filename.split("_")[0].
# Standard ABIDE acquisition parameters — CROSS-CHECK against the deep-research
# TR report before the final training run.
ABIDE_SITE_TR = {
    "Caltech": 2.0, "CMU": 2.0, "KKI": 2.5, "Leuven": 1.6667, "MaxMun": 3.0,
    "NYU": 2.0, "OHSU": 2.5, "Olin": 1.5, "Pitt": 1.5, "SBL": 2.2,
    "SDSU": 2.0, "Stanford": 2.0, "Trinity": 2.0, "UCLA": 3.0, "UM": 2.0,
    "USM": 2.0, "Yale": 2.0,
}
# OASIS-3 per-scan TR was NOT captured (manifest tr=None, raw deleted). Use the
# standard documented value; consistent with T=164 -> 164*2.2 ~= 6 min scans.
OASIS_DEFAULT_TR = 2.2
AOMIC_TR = {"piop1": 0.75, "piop2": 2.0}
HCP_TR = 0.72
ADNI_TR = 3.0                          # single-band (pending Sagi confirmation)


def build_corpus_entries(lab_root=LAB_ROOT,
                         datasets=("HCP", "ABIDE", "OASIS", "AOMIC", "ADNI")):
    """Scan the five dataset dirs -> (flat entry list, name->indices map).

    Each entry: {"dataset", "path", "subject_id", "tr"}. The name->indices map
    feeds ProportionalBatchSampler.
    """
    lab = Path(lab_root)
    entries: list = []
    by_dataset: dict = {}

    def add(name, path, subject_id, tr):
        by_dataset.setdefault(name, []).append(len(entries))
        entries.append({"dataset": name, "path": str(path),
                        "subject_id": subject_id, "tr": float(tr)})

    if "HCP" in datasets:
        for p in sorted(lab.glob("HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt")):
            add("HCP", p, p.parent.name, HCP_TR)
    if "ABIDE" in datasets:
        for p in sorted(lab.glob("ABIDE_data/downsampled/**/*.pt")):
            site = p.name.split("_")[0]
            tr = ABIDE_SITE_TR.get(site)
            if tr is None:
                logger.warning(f"ABIDE site '{site}' unmapped ({p.name}); skipping")
                continue
            add("ABIDE", p, p.stem, tr)
    if "OASIS" in datasets:
        for p in sorted(lab.glob("OASIS3_data/downsampled/*/rest_*.pt")):
            add("OASIS", p, p.parent.name, OASIS_DEFAULT_TR)
    if "AOMIC" in datasets:
        for p in sorted(lab.glob("AOMIC_data/downsampled/*/sub-*/restingstate_downsampled.pt")):
            proto = "piop1" if "piop1" in str(p).lower() else "piop2"
            add("AOMIC", p, p.parent.name, AOMIC_TR[proto])
    if "ADNI" in datasets:
        for p in sorted(lab.glob("ADNI_data/downsampled/*/I*.pt")):
            add("ADNI", p, p.parent.name, ADNI_TR)

    return entries, by_dataset


def write_corpus_manifest(out_path, lab_root=LAB_ROOT,
                          datasets=("HCP", "ABIDE", "OASIS", "AOMIC", "ADNI")):
    """Scan every scan's native T ONCE and write a corpus manifest CSV:
        dataset,path,subject_id,tr,T_native,upsampled_T
    where upsampled_T = round(T_native * tr / TARGET_TR). Built offline (the
    header scan is slow on the network FS); MixedFMRIDataset then reads this
    instead of re-scanning shapes, and uses upsampled_T to drop too-short scans.
    """
    import csv
    entries, _ = build_corpus_entries(lab_root, datasets)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "path", "subject_id", "tr", "T_native", "upsampled_T"])
        for e in entries:
            T = int(_load_mmap(e["path"]).shape[0])
            up = round(T * e["tr"] / TARGET_TR)
            w.writerow([e["dataset"], e["path"], e["subject_id"],
                        e["tr"], T, up])
            n += 1
            if n % 200 == 0:
                logger.info(f"  corpus manifest: {n}/{len(entries)}")
    logger.info(f"corpus manifest written: {n} scans -> {out_path}")
    return out_path


def entries_from_manifest(manifest_path, datasets=("HCP", "ABIDE", "OASIS", "AOMIC", "ADNI"),
                          min_upsampled_t=0):
    """Build (entries, name->indices) from a corpus manifest CSV, keeping only
    rows whose dataset is requested and whose upsampled_T >= min_upsampled_t
    (drops too-short scans so the T_fixed window never needs padding)."""
    import csv
    entries: list = []
    by_dataset: dict = {}
    dropped = 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] not in datasets:
                continue
            if int(row["upsampled_T"]) < min_upsampled_t:
                dropped += 1
                continue
            by_dataset.setdefault(row["dataset"], []).append(len(entries))
            entries.append({"dataset": row["dataset"], "path": row["path"],
                            "subject_id": row["subject_id"], "tr": float(row["tr"])})
    if dropped:
        logger.info(f"manifest: dropped {dropped} scans with upsampled_T < {min_upsampled_t}")
    return entries, by_dataset


def _load_mmap(path):
    """Memory-map a scan lazily (no RAM until sliced). Shape (T,X,Y,Z) or
    (T,1,X,Y,Z)."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _native_window(T, tr_native, t_fixed, target_tr=TARGET_TR):
    """(start, win): a native window of `win` frames spanning t_fixed*target_tr
    seconds, at a random start. If the scan is shorter than the window, take it
    whole (it gets stretched up to t_fixed by _temporal_resample)."""
    win = max(1, round(t_fixed * target_tr / tr_native))
    if T >= win:
        return int(np.random.randint(0, T - win + 1)), win
    return 0, T


def _temporal_resample(clip, n_out):
    """(n_in,1,X,Y,Z) -> (n_out,1,X,Y,Z) by POLYPHASE resampling along time
    (scipy.signal.resample_poly, per Ariel — anti-aliased FIR, the correct tool
    for resampling a band-limited BOLD signal, vs naive linear interpolation).

    Resamples the cropped window from n_in to n_out samples, i.e. from the
    native TR to TARGET_TR (since n_in native frames span ~n_out*TARGET_TR
    seconds). For HCP (n_in == n_out) it's a no-op.
    """
    n_in = clip.shape[0]
    if n_in == n_out:
        return clip
    g = math.gcd(n_out, n_in)
    up, down = n_out // g, n_in // g                  # new_rate/old_rate = n_out/n_in
    arr = clip.contiguous().numpy()                   # (n_in, 1, X, Y, Z)
    out = resample_poly(arr, up, down, axis=0)        # ~n_out along time
    out = torch.from_numpy(np.ascontiguousarray(out)).float()
    if out.shape[0] > n_out:                          # guard off-by-one from ceil
        out = out[:n_out]
    elif out.shape[0] < n_out:
        pad = n_out - out.shape[0]
        out = torch.cat([out, out[-1:].expand(pad, *out.shape[1:])], dim=0)
    return out.contiguous()


def _finalize(clip, t_fixed, target_shape=TARGET_SHAPE):
    """Materialized native window -> (t_fixed,1,*target_shape), z-scored."""
    clip = clip.float()
    if clip.ndim == 4:                         # (n,X,Y,Z) -> add channel
        clip = clip.unsqueeze(1)
    if tuple(clip.shape[-3:]) != tuple(target_shape):
        clip = F.interpolate(clip, size=tuple(target_shape),
                             mode="trilinear", align_corners=False)
    clip = _temporal_resample(clip, t_fixed)
    return _zscore_per_frame(clip)


def compute_t_fixed_max(lab_root=LAB_ROOT,
                        datasets=("HCP", "ABIDE", "OASIS", "AOMIC", "ADNI"),
                        margin=0):
    """Largest T_fixed window (in TARGET_TR frames) that fits EVERY scan with no
    padding = the global minimum upsampled length round(T_native*tr/TARGET_TR).

    Reads every scan's header T (cheap mmap) once. Returns
    (t_fixed_max - margin, per_dataset_min, shortest_scan_entry, per_all_upsampled).
    Use it offline to pick DEFAULT_T_FIXED; a small margin guards the
    round()/edge off-by-one. `per_all_upsampled` (dataset -> list of every scan's
    upsampled T) lets callers compute how many scans a larger T_fixed would drop.
    """
    entries, _ = build_corpus_entries(lab_root, datasets)
    per: dict = {}
    per_all: dict = {}       # dataset -> list of every scan's upsampled T
    g_min, argmin = None, None
    for e in entries:
        T = _load_mmap(e["path"]).shape[0]
        up = round(T * e["tr"] / TARGET_TR)
        d = e["dataset"]
        per[d] = min(per.get(d, up), up)
        per_all.setdefault(d, []).append(up)
        if g_min is None or up < g_min:
            g_min, argmin = up, e
    return max(1, g_min - margin), per, argmin, per_all


class HCPFullScanDataset(Dataset):
    """One HCP subject = one (T, 1, X, Y, Z) tensor reconstructed from windows.pt.

    The on-disk files are
        {hcp_root}/subject_*/MNINonLinear/Results/rfMRI_REST1_LR/windows.pt
    each of shape (N_short, T_short, 1, X, Y, Z). Concatenating every-other
    window (w = 0, 2, 4, ...) reconstructs the full original time series
    (HCP REST1_LR: 1200 frames, T_short=10, stride=5 -> 120 even windows).
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        window_size: int = 10,
        window_stride: int = 5,
        temporal_crop: Optional[int] = None,
        target_shape: Optional[Tuple[int, int, int]] = (45, 54, 45),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        # Default root from env if unset (matches Moriah layout).
        self.hcp_root = Path(root) if root else Path(
            "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
        )
        assert window_size % window_stride == 0, (
            "Cannot reconstruct a full scan without overlap unless "
            "window_size is a multiple of window_stride."
        )
        self.window_size = window_size
        self.window_stride = window_stride
        self.short_step = window_size // window_stride
        # If `temporal_crop` is set, __getitem__ returns a random T=temporal_crop
        # window from the full T=1200 scan. This is used for mixed-dataset
        # training where ADNI is at T=140 — we want HCP scans to also be
        # T=temporal_crop so all datasets share the same temporal length.
        self.temporal_crop = temporal_crop
        # HCP native spatial = (46, 55, 46), ADNI native = (45, 54, 45).
        # Resample HCP -> (45, 54, 45) so Mixed dataset can torch.stack uniformly.
        # The model was already happy with both (conv K=3 S=3 P=0 floors to the
        # same output) but the collate is not.
        self.target_shape = target_shape
        self.transform = transform
        self.target_transform = target_transform

        self.paths = sorted(
            self.hcp_root.glob(
                "subject_*/MNINonLinear/Results/rfMRI_REST1_LR/windows.pt"
            )
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No windows.pt found under {self.hcp_root}."
            )
        logger.info(
            f"HCPFullScanDataset: {len(self.paths)} subjects under {self.hcp_root}"
            + (f" (temporal_crop={temporal_crop})" if temporal_crop else "")
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int) -> torch.Tensor:
        tensor = torch.load(self.paths[idx], map_location="cpu", mmap=True)
        non_overlapping = tensor[:: self.short_step].float()    # (N', T_short, 1, X, Y, Z)
        scan = non_overlapping.reshape(-1, *non_overlapping.shape[2:])  # (T_full, 1, X, Y, Z)
        scan = _zscore_per_frame(scan)
        if self.temporal_crop is not None and scan.shape[0] > self.temporal_crop:
            # Random temporal window of length `temporal_crop`.
            start = int(np.random.randint(0, scan.shape[0] - self.temporal_crop + 1))
            scan = scan[start:start + self.temporal_crop]
        if self.target_shape is not None and tuple(scan.shape[-3:]) != tuple(self.target_shape):
            scan = F.interpolate(
                scan, size=tuple(self.target_shape),
                mode="trilinear", align_corners=False,
            )
        return scan

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = self._load(idx)
        target: Any = 0
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ADNIFullScanDataset(Dataset):
    """One ADNI scan = one (T, 1, X, Y, Z) tensor.

    The shared tensor on disk is (N, X, Y, Z, T) fp32. We mmap it and slice
    + permute one scan at a time so each rank only materialises the scan
    it currently needs.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        target_shape: Optional[Tuple[int, int, int]] = (45, 54, 45),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.adni_path = root or (
            "/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data/all_4d_downsampled.pt"
        )
        self.data = torch.load(
            self.adni_path, weights_only=True, map_location="cpu", mmap=True,
        )
        if self.data.ndim != 5:
            raise ValueError(
                f"Expected 5D ADNI tensor, got shape {tuple(self.data.shape)}"
            )
        non_batch = list(self.data.shape[1:])
        self.t_axis = 1 + int(np.argmax(non_batch))
        # ADNI native spatial = (46, 55, 46), HCP / model target = (45, 54, 45).
        # Resample to target_shape via trilinear, matching what probe_adni.py does.
        self.target_shape = target_shape
        self.transform = transform
        self.target_transform = target_transform
        logger.info(
            f"ADNIFullScanDataset: {self.data.shape[0]} scans, shape={tuple(self.data.shape)}, t_axis={self.t_axis}, target_shape={target_shape}"
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def _load(self, idx: int) -> torch.Tensor:
        if self.t_axis == 1:
            scan = self.data[idx]
        elif self.t_axis == 4:
            scan = self.data[idx].permute(3, 0, 1, 2)
        else:
            perm = [self.t_axis - 1] + [
                i for i in range(self.data[idx].ndim) if i != self.t_axis - 1
            ]
            scan = self.data[idx].permute(*perm)
        scan = scan.contiguous().float().unsqueeze(1)         # (T, 1, X, Y, Z)
        if self.target_shape is not None and tuple(scan.shape[-3:]) != tuple(self.target_shape):
            scan = F.interpolate(
                scan, size=tuple(self.target_shape),
                mode="trilinear", align_corners=False,
            )
        return _zscore_per_frame(scan)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = self._load(idx)
        target: Any = 0
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class MixedFMRIDataset(Dataset):
    """All five fMRI cohorts (HCP / ABIDE / OASIS-3 / AOMIC / ADNI) as one dataset.

    Each __getitem__:
      1. memory-maps a native (T,X,Y,Z) scan,
      2. slices a native window spanning T_fixed * 0.72 s of brain activity
         (crop FIRST, on the mmap, so a 524 MB HCP file never fully loads),
      3. materializes only that window and resamples it to exactly T_fixed
         frames at the common TR = 0.72 s (temporal harmonization across the
         heterogeneous native TRs: HCP 0.72, ABIDE per-site, OASIS 2.2,
         AOMIC 0.75/2.0, ADNI 3.0),
      4. per-frame z-scores -> (T_fixed, 1, 45, 54, 45).

    `self.dataset_indices` (name -> global indices) is exposed so a
    ProportionalBatchSampler can build per-dataset balanced batches.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        t_fixed: int = DEFAULT_T_FIXED,
        temporal_crop: Optional[int] = None,        # legacy alias for t_fixed
        datasets: Tuple[str, ...] = ("HCP", "ABIDE", "OASIS", "AOMIC", "ADNI"),
        manifest: Optional[str] = None,             # corpus_manifest.csv; auto if present
        drop_short: bool = True,                    # drop scans whose window needs padding
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **_ignored,
    ):
        if temporal_crop is not None:
            t_fixed = temporal_crop
        self.t_fixed = int(t_fixed)
        lab_root = root or LAB_ROOT

        # Prefer the corpus manifest (fast + carries T_native so we can drop
        # too-short scans). Falls back to globbing if no manifest is found.
        man = Path(manifest) if manifest else Path(lab_root) / DEFAULT_MANIFEST
        if man.exists():
            min_t = self.t_fixed if drop_short else 0
            self.entries, self.dataset_indices = entries_from_manifest(
                man, datasets, min_upsampled_t=min_t
            )
            src = f"manifest {man.name}" + (f" (drop_short<{min_t})" if drop_short else "")
        else:
            self.entries, self.dataset_indices = build_corpus_entries(lab_root, datasets)
            src = "glob (no manifest; short scans will be stretched, not dropped)"
            if drop_short:
                logger.warning(
                    f"drop_short requested but no manifest at {man} — short scans "
                    "cannot be filtered without T_native. Build it with "
                    "write_corpus_manifest()."
                )

        if not self.entries:
            raise FileNotFoundError(
                f"No scans found under {lab_root} for datasets={datasets}"
            )
        self.transform = transform
        self.target_transform = target_transform
        counts = {k: len(v) for k, v in self.dataset_indices.items()}
        logger.info(
            f"MixedFMRIDataset: {len(self.entries)} scans  T_fixed={self.t_fixed}  "
            f"target_TR={TARGET_TR}s  src={src}  per-dataset={counts}"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, idx: int) -> torch.Tensor:
        e = self.entries[idx]
        scan = _load_mmap(e["path"])                       # (T,X,Y,Z) lazy
        T = scan.shape[0]
        start, win = _native_window(T, e["tr"], self.t_fixed)
        clip = scan[start:start + win].clone()             # materialize ONLY the window
        return _finalize(clip, self.t_fixed)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = self._load(idx)
        target: Any = 0
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ProportionalBatchSampler(Sampler):
    """Yield batches with a fixed per-dataset composition.

    Default quota (batch of 16, ~proportional to dataset sizes):
        HCP 4, ABIDE 4, OASIS 4, ADNI 3, AOMIC 1.
    Each dataset's indices are shuffled and consumed without replacement; a
    dataset that runs out within an epoch is reshuffled and cycled (smaller
    datasets cycle more often). Tokens within a batch are shuffled so the
    dataset order is not fixed.

    Use with DataLoader(dataset, batch_sampler=sampler). For DDP, pass
    rank/world_size so each rank draws a disjoint stream of batches.
    """

    DEFAULT_QUOTA = {"HCP": 4, "ABIDE": 4, "OASIS": 4, "ADNI": 3, "AOMIC": 1}

    def __init__(self, dataset_indices, quota=None, *, batches_per_epoch=None,
                 seed=0, rank=0, world_size=1):
        self.dataset_indices = {k: list(v) for k, v in dataset_indices.items() if v}
        self.quota = {k: q for k, q in (quota or self.DEFAULT_QUOTA).items()
                      if k in self.dataset_indices and q > 0}
        if not self.quota:
            raise ValueError(
                f"No dataset matches quota keys. Have {list(self.dataset_indices)}, "
                f"quota {list((quota or self.DEFAULT_QUOTA))}"
            )
        self.batch_size = sum(self.quota.values())
        self.seed = seed
        self.rank = rank
        self.world_size = max(1, world_size)
        self.epoch = 0
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
            order = torch.randperm(len(idxs), generator=g).tolist()
            pools[name] = [idxs[i] for i in order]
            ptr[name] = 0
        for _ in range(self.batches_per_epoch):
            batch = []
            for name, q in self.quota.items():
                pool, p = pools[name], ptr[name]
                for _ in range(q):
                    if p >= len(pool):                     # epoch-local reshuffle + cycle
                        order = torch.randperm(len(pool), generator=g).tolist()
                        pool = [pool[i] for i in order]
                        pools[name] = pool
                        p = 0
                    batch.append(pool[p])
                    p += 1
                ptr[name] = p
            order = torch.randperm(len(batch), generator=g).tolist()
            yield [batch[i] for i in order]


# ----------------------------------------------------------------------
# Multi-crop (3D spatial only, no photometric augs).
#
# Constructor signature matches `DataAugmentationDINO` so `do_train` can
# swap classes via `cfg.train.fmri_augmentation` without further changes.
# `global_crops_size` / `local_crops_size` are accepted to keep the
# signature identical but are not used: a 3D crop is resampled back to
# the input volume's (X, Y, Z) so all crops share one fixed token grid.
# ----------------------------------------------------------------------

class MultiCrop3D:
    """3D spatial multi-crop for fMRI volumes."""

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,        # accepted, unused (kept for API parity)
        local_crops_size=96,          # accepted, unused (kept for API parity)
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

        logger.info("###################################")
        logger.info("Using fMRI multi-crop parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info("###################################")

    @staticmethod
    def _random_resized_crop(scan: torch.Tensor, scale) -> torch.Tensor:
        T, C, X, Y, Z = scan.shape
        frac = float(np.random.uniform(*scale))
        cx, cy, cz = max(1, int(X * frac)), max(1, int(Y * frac)), max(1, int(Z * frac))
        sx = int(np.random.randint(0, max(X - cx, 1)))
        sy = int(np.random.randint(0, max(Y - cy, 1)))
        sz = int(np.random.randint(0, max(Z - cz, 1)))
        sub = scan[:, :, sx:sx + cx, sy:sy + cy, sz:sz + cz]
        return F.interpolate(sub, size=(X, Y, Z),
                             mode="trilinear", align_corners=False)

    def __call__(self, scan: torch.Tensor) -> dict:
        global_crops = [
            self._random_resized_crop(scan, self.global_crops_scale)
            for _ in range(2)
        ]
        local_crops = [
            self._random_resized_crop(scan, self.local_crops_scale)
            for _ in range(self.local_crops_number)
        ]
        return {
            "global_crops":         global_crops,
            "global_crops_teacher": global_crops,
            "local_crops":          local_crops,
            "offsets":              (),
        }
