"""Offline corpus tools — NOT part of the training runtime.

Run once (or whenever the corpus changes) to prepare the data:
  write_corpus_manifest : read every scan's native length once -> corpus_manifest.csv
  compute_t_fixed_max   : find the largest T_fixed that fits every scan (to pick T=270)

MixedFMRIDataset (fmri_data.py) reads the manifest at training time; it never calls
anything here. Kept in a separate module so fmri_data.py stays runtime-only.
"""

import csv
import logging
from pathlib import Path

from .fmri_const import LAB_ROOT, TARGET_TR, CORPUS_DATASETS
from .fmri_data import build_corpus_entries, _load_mmap

logger = logging.getLogger("dinov2")


def write_corpus_manifest(out_path, lab_root=LAB_ROOT, datasets=CORPUS_DATASETS):
    """Offline: scan every scan's native T once and write the corpus manifest CSV.
    MixedFMRIDataset then reads this instead of re-scanning shapes, using
    upsampled_T to drop too-short scans. Returns the written path.

    Example row (upsampled_T = round(T_native * tr / TARGET_TR)):
      dataset,path,subject_id,tr,T_native,upsampled_T
      ADNI,/.../I123456.pt,sub-4123,3.0,140,583        # 140 frames @ 3.0s -> 583 @ 0.72s
      HCP,/.../subject_100206/...pt,subject_100206,0.72,1200,1200   # already 0.72s -> unchanged
    """
    entries = build_corpus_entries(lab_root, datasets)
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


def compute_t_fixed_max(lab_root=LAB_ROOT, datasets=CORPUS_DATASETS, margin=0):
    """Offline: largest T_fixed that fits EVERY scan with no padding = global min
    upsampled length. Returns (t_fixed_max - margin, per_dataset_min, shortest_entry,
    per_dataset_all_upsampled). Use to pick DEFAULT_T_FIXED."""
    entries = build_corpus_entries(lab_root, datasets)
    per, per_all, g_min, argmin = {}, {}, None, None
    for e in entries:
        up = round(_load_mmap(e["path"]).shape[0] * e["tr"] / TARGET_TR)
        d = e["dataset"]
        per[d] = min(per.get(d, up), up)
        per_all.setdefault(d, []).append(up)
        if g_min is None or up < g_min:
            g_min, argmin = up, e
    return max(1, g_min - margin), per, argmin, per_all
