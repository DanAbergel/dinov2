"""70:30 train/test split by subject (leakage-free).

In-corpus cohorts (ADNI/ABIDE/HCP/OASIS) use subject_split.json — the 30% test
subjects were held out of pretraining, so the encoder never saw them. External
cohorts (never pretrained on) get a deterministic random 70:30 by subject.
"""

import json
from pathlib import Path

import numpy as np

from dinov2.data.fmri_data import LAB_ROOT, DEFAULT_SPLIT

LAB = Path(LAB_ROOT)


def split_masks(name, subjects):
    """Return (train_mask, test_mask) boolean arrays aligned with `subjects`."""
    sp = json.loads((LAB / DEFAULT_SPLIT).read_text())["datasets"]
    if name in sp:                                   # in-corpus -> use the holdout split
        m = {s: k for k, subs in sp[name].items() for s in subs}
        return (np.array([m.get(s) == "train" for s in subjects]),
                np.array([m.get(s) in ("val", "test") for s in subjects]))
    uniq = sorted(set(subjects))                     # external -> deterministic random 70:30
    np.random.RandomState(0).shuffle(uniq)
    train = set(uniq[:int(0.7 * len(uniq))])
    tr = np.array([s in train for s in subjects])
    return tr, ~tr
