"""Constants for the multi-source fMRI corpus.

Paths, per-dataset native TR, and the temporal-harmonization targets used by
fmri_data.py (training) and by the downstream probe. Every scan on disk is a
(T, X, Y, Z) float32 tensor at a fixed spatial resolution of (45, 54, 45); the
only thing that differs across datasets is the acquisition TR (seconds between
volumes), which we harmonize to TARGET_TR with a fixed T_FIXED-frame window.
"""

LAB_ROOT = "/sci/labs/arieljaffe/dan.abergel1"
TARGET_TR = 0.72                       # common TR after harmonization (HCP native)
TARGET_SHAPE = (45, 54, 45)            # fixed spatial resolution (X, Y, Z)

# T_fixed = 270 frames @ 0.72s = 194.4s window. At the knee of the window-vs-scans
# trade-off: 1 frame under ABIDE's min upsampled length (271) so all of ABIDE is
# kept, dropping only the 2 short OASIS outliers (< 270).
DEFAULT_T_FIXED = 270
DEFAULT_MANIFEST = "corpus_manifest.csv"   # under LAB_ROOT; auto-used if present
DEFAULT_SPLIT = "subject_split.json"       # under LAB_ROOT; auto-used if present

# The five pretraining sources.
CORPUS_DATASETS = ("HCP", "ABIDE", "OASIS", "AOMIC", "ADNI")

# Holdout: the test SUBJECTS of these are excluded from SSL pretraining (no
# leakage) so probes evaluate on subjects the encoder never saw. AOMIC stays
# pretraining-only (no probe) -> kept whole.
HOLDOUT_DATASETS = ("ADNI", "ABIDE", "OASIS", "HCP")

# Training-time corpus filters. Kept here (not as MixedFMRIDataset arguments) so the
# dataset takes almost no args — change these here to change training behavior.
PRETRAIN_SPLITS = ("train",)   # which splits enter pretraining (test is held out)
DROP_SHORT = True              # drop scans too short to fill a T_FIXED window

# Native TR (seconds) per dataset. ABIDE varies per site (site = filename.split("_")[0]).
ABIDE_SITE_TR = {
    "Caltech": 2.0, "CMU": 2.0, "KKI": 2.5, "Leuven": 1.6667, "MaxMun": 3.0,
    "NYU": 2.0, "OHSU": 2.5, "Olin": 1.5, "Pitt": 1.5, "SBL": 2.2,
    "SDSU": 2.0, "Stanford": 2.0, "Trinity": 2.0, "UCLA": 3.0, "UM": 2.0,
    "USM": 2.0, "Yale": 2.0,
}
OASIS_DEFAULT_TR = 2.2                  # per-scan TR not captured; documented value
AOMIC_TR = {"piop1": 0.75, "piop2": 2.0}
HCP_TR = 0.72
ADNI_TR = 3.0
