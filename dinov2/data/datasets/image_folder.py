# Generic folder-of-folders image dataset (root/<class>/*.jpg).
#
# Added ON TOP of official DINOv2 to run the Imagenette control test: official
# DINOv2 only ships ImageNet / ImageNet22k + the cell datasets, none of which
# read a plain image folder. This class fills that gap and ingests images exactly
# like the ImageNet class (raw bytes -> ImageDataDecoder), so the training path
# (model, losses, DataAugmentationDINO, sampler, loop) stays the official one.

import logging

from torchvision.datasets import ImageFolder as _TorchvisionImageFolder

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


class ImageFolder(ExtendedVisionDataset):
    """Plain image folder. `root` holds one sub-directory per class, each with
    image files (the torchvision ImageFolder layout = Imagenette/ImageNet train)."""

    def __init__(
        self,
        *,
        root: str,
        transforms=None,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        # torchvision ImageFolder only scans the tree here (builds the
        # (path, class_index) list); it reads no pixels until access.
        self._samples = _TorchvisionImageFolder(root).samples
        if not self._samples:
            raise RuntimeError(f"ImageFolder: no images found under {root}")
        logger.info(f"ImageFolder: {len(self._samples):,d} images at {root}")

    def get_image_data(self, index: int) -> bytes:
        path, _ = self._samples[index]
        with open(path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> int:
        return self._samples[index][1]

    def __len__(self) -> int:
        return len(self._samples)
