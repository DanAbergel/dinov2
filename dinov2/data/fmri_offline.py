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

from .fmri_const import (LAB_ROOT, TARGET_TR, CORPUS_DATASETS, HCP_TR,
                         ABIDE_SITE_TR, OASIS_DEFAULT_TR, AOMIC_TR, ADNI_TR)
from .fmri_data import _load_mmap

logger = logging.getLogger("dinov2")


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
    """Discover every scan of the requested datasets on disk (OFFLINE step).

    Args:
      lab_root : root dir holding the <DATASET>_data/downsampled/ folders.
      datasets : which cohorts to include (subset of DATASET_SOURCES keys).
    Returns:
      `entries`: a flat list of {dataset, path, subject_id, tr}, one dict per scan.
      write_corpus_manifest reads each scan's length and writes the CSV from this.

    Example:
      [
        {"dataset": "HCP",   "path": ".../subject_100206/...pt", "subject_id": "subject_100206", "tr": 0.72},
        {"dataset": "ABIDE", "path": ".../NYU_0051091.pt",       "subject_id": "NYU_0051091",    "tr": 2.0 },
        ...
      ]
    """
    entries = []
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
            entries.append({"dataset": name, "path": str(p),
                            "subject_id": src["subject"](p), "tr": float(tr)})
    return entries


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
