"""Self-contained probe labels module.

Extracted from the FAIR repo (src/config.py + src/baselines/utils.py) so the
probe scripts in this repo don't depend on a sibling repo being present.
Contains:
  - HCP / ADNI paths
  - HCP_LABELS / ADNI_LABELS schemas
  - CV constants (N_SPLITS, RANDOM_STATE)
  - load_adni_labels  : build a row-aligned labels DataFrame for the ADNI tensor
  - get_label_array   : extract one column from a labels DataFrame as a filtered array

No third-party deps beyond numpy / pandas.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PATHS -- HCP
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
# HCP_YA_subjects*.csv is the per-subject labels file. The filename has a
# timestamp suffix in some setups; pick whatever matches the glob.
_csv_matches = sorted(HCP_ROOT.glob("data/HCP_YA_subjects*.csv"))
HCP_SUBJECTS_CSV = _csv_matches[0] if _csv_matches else (
    HCP_ROOT / "data" / "HCP_YA_subjects.csv"
)

# =============================================================================
# PATHS -- ADNI
# =============================================================================
ADNI_ROOT = Path("/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data")
ADNI_INDEX_JSON = ADNI_ROOT / "index_to_name.json"
ADNI_LABELS_JSON = ADNI_ROOT / "imageID_to_labels.json"

# =============================================================================
# CV SETTINGS
# =============================================================================
N_SPLITS = 5
RANDOM_STATE = 42

# =============================================================================
# LABELS
# =============================================================================
HCP_LABELS = {
    "Sex": {
        "column": "Gender",
        "type": "classification",
        "transform": lambda x: 1 if x == "M" else 0,
        "scoring": "roc_auc",
        "class_names": {1: "Male", 0: "Female"},
    },
    "Age": {
        "column": "Age_in_Yrs",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "BrainVol": {
        "column": "FS_IntraCranial_Vol",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "GrayMatterVol": {
        "column": "FS_TotCort_GM_Vol",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "FluidIntel": {
        "column": "PMAT24_A_CR",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "ProcSpeed": {
        "column": "ProcSpeed_AgeAdj",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "WorkingMem": {
        "column": "ListSort_AgeAdj",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
}

ADNI_LABELS = {
    "Sex": {
        "column": "Sex_Binary",
        "type": "classification",
        "transform": int,
        "scoring": "roc_auc",
    },
    "Age": {
        "column": "Age",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "MMSE": {
        "column": "MMSE Total Score",
        "type": "regression",
        "transform": float,
        "scoring": "neg_mean_absolute_error",
    },
    "CDR": {
        "column": "CDR_Binary",
        "type": "classification",
        "transform": int,
        "scoring": "roc_auc",
    },
    "Degradation1Y": {
        "column": "degradation_binary_1year",
        "type": "classification",
        "transform": int,
        "scoring": "roc_auc",
    },
    "Degradation2Y": {
        "column": "degradation_binary_2years",
        "type": "classification",
        "transform": int,
        "scoring": "roc_auc",
    },
    "Degradation3Y": {
        "column": "degradation_binary_3years",
        "type": "classification",
        "transform": int,
        "scoring": "roc_auc",
    },
}

# Backward compatibility: default to HCP.
LABELS = HCP_LABELS


# =============================================================================
# LABEL UTILITIES
# =============================================================================

def load_adni_labels(index_to_name_path, labels_path) -> pd.DataFrame:
    """Build a row-aligned labels DataFrame for the ADNI 4D tensor.

    ADNI scans are stored in one big tensor of shape (N, X, Y, Z, T). Scan i is
    addressed by its integer index. Two JSON files connect the integer index to
    clinical labels:
        index_to_name.json    :  "0" -> {"image_id": "I123", "subject_id": "002_S_0413", ...}
        imageID_to_labels.json:  "I123" -> {AGE: 76.5, MMSE: 28, CDR: 0.5, ...}
    This function joins them. Missing labels appear as NaN.
    """
    with open(index_to_name_path) as f:
        index_to_name = json.load(f)
    with open(labels_path) as f:
        image_labels = json.load(f)

    rows = []
    for idx in sorted(index_to_name.keys(), key=int):
        entry = index_to_name[idx]
        image_id = entry["image_id"]
        row = {"subject_id": entry["subject_id"], "image_id": image_id}
        if image_id in image_labels:
            row.update(image_labels[image_id])
        rows.append(row)

    return pd.DataFrame(rows)


def get_label_array(labels_df: pd.DataFrame, label_name: str, labels_dict: dict | None = None):
    """Extract one column from labels_df as a (NaN-filtered, transformed) array.

    Returns:
        valid_indices : list[int] -- rows of labels_df that survived the filter.
        y             : np.ndarray (len(valid_indices),) -- transformed values.

    The caller usually has a separate features array X aligned with labels_df;
    after filtering, X must be subset to valid_indices so X[valid_indices] and y
    have the same length and same row order.
    """
    label_cfg = (labels_dict or LABELS)[label_name]
    col = label_cfg["column"]
    transform = label_cfg["transform"]

    valid_indices = []
    y_list = []

    for i, (_, row) in enumerate(labels_df.iterrows()):
        val = row.get(col)
        if pd.isna(val):
            continue
        try:
            y_list.append(transform(val))
            valid_indices.append(i)
        except (ValueError, TypeError):
            continue

    return valid_indices, np.array(y_list)
