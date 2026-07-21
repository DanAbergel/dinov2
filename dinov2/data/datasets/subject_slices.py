# Groups a brain2d-style folder (root/<class>/subject_<id>_..._zNNN.png) BY SUBJECT.
#
# One sample = one SUBJECT (not one slice). __getitem__ returns the list of that subject's
# slice paths; the paired transform (MultiSliceAugmentationDINO) then samples a few of them
# so that the 2 global crops and the local crops each come from a DIFFERENT slice of the
# SAME brain. This forces the model to be invariant to slice position and to rely on the
# subject's own features to align the views — instead of just discriminating slice height.
import glob
import logging
import os
import re

from torch.utils.data import Dataset

logger = logging.getLogger("dinov2")
_SUBJ = re.compile(r"subject_(\d+)")


class SubjectSliceFolder(Dataset):
    """root/<class>/subject_<id>_*.png grouped by subject id. len = number of subjects.
    __getitem__ passes the subject's list of slice PATHS to the transform (which samples,
    loads and augments a few of them)."""

    def __init__(self, *, root, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        by_subject = {}
        for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
            m = _SUBJ.search(os.path.basename(p))
            if m:
                by_subject.setdefault(m.group(1), []).append(p)
        self.subjects = [(sid, paths) for sid, paths in sorted(by_subject.items())]
        if not self.subjects:
            raise RuntimeError(f"SubjectSliceFolder: no subject_<id> PNGs found under {root}")
        n_slices = sum(len(p) for _, p in self.subjects)
        logger.info(
            f"SubjectSliceFolder: {len(self.subjects)} subjects, {n_slices} slices "
            f"(~{n_slices / len(self.subjects):.1f}/subject) at {root}"
        )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        sid, paths = self.subjects[index]
        out = self.transform(paths) if self.transform is not None else paths
        target = self.target_transform(sid) if self.target_transform is not None else sid
        return out, target
