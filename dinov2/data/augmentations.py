# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import re

from PIL import Image
from torchvision import transforms

_Z_RE = re.compile(r"_z(\d+)")


def _slice_z(path):
    m = _Z_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 0

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        if os.environ.get("DINO_LOCAL_EQ_GLOBAL") == "1":
            # DIAGNOSTIC (not a training improvement): make every local crop pixel-identical
            # to global_crop_1 — no independent RandomResizedCrop / flip / color / blur. This
            # tests whether dino_local stays high only because the local views differ from the
            # teacher's globals. With identical views, dino_local should collapse toward its
            # floor (aligning an image with itself through the EMA teacher). Risk: trivial
            # objective / representation collapse, so judge by the probe, not the loss.
            local_crops = [global_crop_1 for _ in range(self.local_crops_number)]
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class MultiSliceAugmentationDINO(DataAugmentationDINO):
    """Same crops/augmentations as DataAugmentationDINO, but each view comes from a DIFFERENT
    randomly-sampled slice of the SAME subject. Input is the LIST of that subject's slice
    PATHS (from SubjectSliceFolder). The 2 global crops = 2 different slices; the N local crops
    = N other different slices. Position is randomised between the views to be aligned, so the
    model can't use slice height as a shortcut and must rely on subject features. The batch
    still contains slices at many positions, so the teacher target stays peaked (loss active)."""

    def __call__(self, paths):
        need = 2 + self.local_crops_number
        window = int(os.environ.get("DINO_SLICE_WINDOW", "0"))
        if window > 0 and len(paths) >= need:
            # ADJACENT mode: sample all crops from a random consecutive window of positions,
            # so the paired slices are NEARBY (share anatomy) -> alignable without collapse.
            ordered = sorted(paths, key=_slice_z)
            w = max(need, min(window, len(ordered)))
            start = random.randint(0, len(ordered) - w)
            chosen = random.sample(ordered[start:start + w], need)
        elif len(paths) >= need:
            chosen = random.sample(paths, need)          # RANDOM mode: any slices of the subject
        else:
            chosen = [random.choice(paths) for _ in range(need)]
        imgs = [Image.open(p).convert("RGB") for p in chosen]

        output = {}
        global_crop_1 = self.global_transfo1(self.geometric_augmentation_global(imgs[0]))
        global_crop_2 = self.global_transfo2(self.geometric_augmentation_global(imgs[1]))
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        output["local_crops"] = [
            self.local_transfo(self.geometric_augmentation_local(imgs[2 + i]))
            for i in range(self.local_crops_number)
        ]
        output["offsets"] = ()
        return output
