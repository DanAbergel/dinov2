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

class SliceNeighborsFolder(Dataset):
    """One sample = one image; __getitem__ hands the transform (ordered_group_paths, index).
    Grouping/ordering set by env DINO_NEIGHBOR_MODE:
      "z" (default): group by subject, order by _z -> neighbors in POSITION or TIME-as-z
          (root/<class>/subject_<id>_z<NNN>.png)
      "slice_time" : group by (subject, slice z), order by _t -> neighbors in TIME per slice
          (root/<class>/subject_<id>_z<ZZZ>_t<TTT>.png), i.e. many slices/subject, each a time series."""

    def __init__(self, *, root, transforms=None, transform=None, target_transform=None):
        self.transform = transform
        if os.environ.get("DINO_NEIGHBOR_MODE", "z") == "slice_time":
            group_re = re.compile(r"subject_(\d+)_z(\d+)")   # group = subject + slice
            order_re = re.compile(r"_t(\d+)")                # order by timepoint
        else:
            group_re = re.compile(r"subject_(\d+)")          # group = subject
            order_re = re.compile(r"_z(\d+)")                # order by z (position, or time named as z)

        def order_key(p):
            m = order_re.search(os.path.basename(p))
            return int(m.group(1)) if m else 0

        groups = {}
        for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
            m = group_re.search(os.path.basename(p))
            if m:
                groups.setdefault("_".join(m.groups()), []).append(p)
        self.index = []                                   # list of (ordered_paths, idx_in_group)
        for _k, ps in sorted(groups.items()):
            ps = sorted(ps, key=order_key)
            for i in range(len(ps)):
                self.index.append((ps, i))
        if not self.index:
            raise RuntimeError(f"SliceNeighborsFolder: no matching PNGs under {root} (mode={os.environ.get('DINO_NEIGHBOR_MODE', 'z')})")

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
