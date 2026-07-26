# fMRI add-on (not upstream DINOv2): neighbor-slice SSL.
#
# One sample = one axial slice z. Global crops = 2 augmented views of z (STANDARD DINO, so the
# global<->global pair stays a healthy same-slice pair — this is what avoids the collapse we saw
# with different-slice global pairs). Local crops = the N neighboring slices z + k*stride
# (k in [-N/2 .. N/2-1]), WITHOUT any augmentation. The student sees neighbor slices and must
# match the teacher on z -> invariance to a small slice shift on the same brain.
#
# Dataset root = a DENSE pool (consecutive z): root/<class>/subject_<id>_z<NNN>.png
# Env: DINO_NEIGHBOR_STRIDE (default 1) = spacing between the neighbor slices.

import glob
import os
import re

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..augmentations import DataAugmentationDINO

_SUBJ = re.compile(r"subject_(\d+)")
_Z = re.compile(r"_z(\d+)")


def _zpos(p):
    m = _Z.search(os.path.basename(p))
    return int(m.group(1)) if m else 0


class SliceNeighborsFolder(Dataset):
    """root/<class>/subject_<id>_z<NNN>.png (dense, consecutive z). One sample = one slice;
    __getitem__ hands the transform (ordered_slice_paths_of_that_subject, index_of_this_slice)."""

    def __init__(self, *, root, transforms=None, transform=None, target_transform=None):
        self.transform = transform
        by_subject = {}
        for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
            m = _SUBJ.search(os.path.basename(p))
            if m:
                by_subject.setdefault(m.group(1), []).append(p)
        self.index = []                                   # list of (ordered_paths, idx_in_subject)
        for _sid, ps in sorted(by_subject.items()):
            ps = sorted(ps, key=_zpos)
            for i in range(len(ps)):
                self.index.append((ps, i))
        if not self.index:
            raise RuntimeError(f"SliceNeighborsFolder: no subject_<id>_z<NN>.png under {root}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ordered, idx = self.index[i]
        out = self.transform((ordered, idx)) if self.transform is not None else (ordered, idx)
        return out, 0                                     # SSL: dummy target


class NeighborSliceAugmentation(DataAugmentationDINO):
    """Global = standard 2 augmented views of slice z. Local = N neighbor slices, no augmentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = max(1, int(os.environ.get("DINO_NEIGHBOR_STRIDE", "1")))
        # local = plain neighbor slice: resize + normalize only (NO crop/flip/color/blur)
        self.local_plain = transforms.Compose(
            [
                transforms.Resize(
                    (self.local_crops_size, self.local_crops_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                self.normalize,
            ]
        )

    def __call__(self, sample):
        paths, center = sample
        n = len(paths)
        img_z = Image.open(paths[center]).convert("RGB")

        # global crops: STANDARD DINO on slice z (unchanged)
        gc1 = self.global_transfo1(self.geometric_augmentation_global(img_z))
        gc2 = self.global_transfo2(self.geometric_augmentation_global(img_z))
        output = {"global_crops": [gc1, gc2], "global_crops_teacher": [gc1, gc2]}

        # local crops: neighbor slices z + k*stride, k in [-L/2 .. L/2-1], NO augmentation
        L = self.local_crops_number
        local = []
        for k in range(L):
            off = (k - L // 2) * self.stride
            j = min(max(center + off, 0), n - 1)          # clamp at the brain edges
            local.append(self.local_plain(Image.open(paths[j]).convert("RGB")))
        output["local_crops"] = local
        output["offsets"] = ()
        return output
