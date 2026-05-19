# fMRI data utilities — drop-in for DINOv2's official pipeline.
#
# Two Dataset classes (HCP, ADNI). Each returns ONE full scan as a
# (T, 1, X, Y, Z) z-scored tensor. The temporal dimension is kept intact
# and reduced by the PatchEmbed3DPlus1D layer.
#
# One MultiCrop3D class — 3D spatial multi-crop only, no photometric augs.
# Its constructor matches `DataAugmentationDINO` so it drops into
# `do_train` as a one-line replacement (via cfg.train.fmri_augmentation).
# `global_crops_size` / `local_crops_size` are accepted but ignored: fMRI
# crops are resized back to the input volume shape (X, Y, Z) so the ViT
# pos_embed table stays one fixed size.
#
# No colour jitter / blur / solarize / flip / noise / ImageNet normalize —
# fMRI volumes are already z-scored.

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


logger = logging.getLogger("dinov2")


# ----------------------------------------------------------------------
# Datasets — same `transform=` / `target_transform=` API as ImageNet so
# they plug into `make_dataset` and `do_train` unchanged.
# ----------------------------------------------------------------------

def _zscore_per_frame(scan: torch.Tensor) -> torch.Tensor:
    """Per-frame z-score on (T, 1, X, Y, Z)."""
    mean = scan.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = scan.std(dim=(1, 2, 3, 4), keepdim=True)
    return torch.where(
        std > 1e-6,
        (scan - mean) / std.clamp_min(1e-6),
        torch.zeros_like(scan),
    )


class HCPFullScanDataset(Dataset):
    """One HCP subject = one (T, 1, X, Y, Z) tensor reconstructed from windows.pt.

    The on-disk files are
        {hcp_root}/subject_*/MNINonLinear/Results/rfMRI_REST1_LR/windows.pt
    each of shape (N_short, T_short, 1, X, Y, Z). Concatenating every-other
    window (w = 0, 2, 4, ...) reconstructs the full original time series
    (HCP REST1_LR: 1200 frames, T_short=10, stride=5 -> 120 even windows).
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        window_size: int = 10,
        window_stride: int = 5,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        # Default root from env if unset (matches Moriah layout).
        self.hcp_root = Path(root) if root else Path(
            "/sci/labs/arieljaffe/dan.abergel1/HCP_data"
        )
        assert window_size % window_stride == 0, (
            "Cannot reconstruct a full scan without overlap unless "
            "window_size is a multiple of window_stride."
        )
        self.window_size = window_size
        self.window_stride = window_stride
        self.short_step = window_size // window_stride
        self.transform = transform
        self.target_transform = target_transform

        self.paths = sorted(
            self.hcp_root.glob(
                "subject_*/MNINonLinear/Results/rfMRI_REST1_LR/windows.pt"
            )
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No windows.pt found under {self.hcp_root}."
            )
        logger.info(f"HCPFullScanDataset: {len(self.paths)} subjects under {self.hcp_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int) -> torch.Tensor:
        tensor = torch.load(self.paths[idx], map_location="cpu", mmap=True)
        non_overlapping = tensor[:: self.short_step].float()    # (N', T_short, 1, X, Y, Z)
        scan = non_overlapping.reshape(-1, *non_overlapping.shape[2:])  # (T_full, 1, X, Y, Z)
        return _zscore_per_frame(scan)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = self._load(idx)
        target: Any = 0
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ADNIFullScanDataset(Dataset):
    """One ADNI scan = one (T, 1, X, Y, Z) tensor.

    The shared tensor on disk is (N, X, Y, Z, T) fp32. We mmap it and slice
    + permute one scan at a time so each rank only materialises the scan
    it currently needs.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.adni_path = root or (
            "/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data/all_4d_downsampled.pt"
        )
        self.data = torch.load(
            self.adni_path, weights_only=True, map_location="cpu", mmap=True,
        )
        if self.data.ndim != 5:
            raise ValueError(
                f"Expected 5D ADNI tensor, got shape {tuple(self.data.shape)}"
            )
        non_batch = list(self.data.shape[1:])
        self.t_axis = 1 + int(np.argmax(non_batch))
        self.transform = transform
        self.target_transform = target_transform
        logger.info(
            f"ADNIFullScanDataset: {self.data.shape[0]} scans, shape={tuple(self.data.shape)}, t_axis={self.t_axis}"
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def _load(self, idx: int) -> torch.Tensor:
        if self.t_axis == 1:
            scan = self.data[idx]
        elif self.t_axis == 4:
            scan = self.data[idx].permute(3, 0, 1, 2)
        else:
            perm = [self.t_axis - 1] + [
                i for i in range(self.data[idx].ndim) if i != self.t_axis - 1
            ]
            scan = self.data[idx].permute(*perm)
        scan = scan.contiguous().float().unsqueeze(1)
        return _zscore_per_frame(scan)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = self._load(idx)
        target: Any = 0
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


# ----------------------------------------------------------------------
# Multi-crop (3D spatial only, no photometric augs).
#
# Constructor signature matches `DataAugmentationDINO` so `do_train` can
# swap classes via `cfg.train.fmri_augmentation` without further changes.
# `global_crops_size` / `local_crops_size` are accepted to keep the
# signature identical but are not used: a 3D crop is resampled back to
# the input volume's (X, Y, Z) so all crops share one fixed token grid.
# ----------------------------------------------------------------------

class MultiCrop3D:
    """3D spatial multi-crop for fMRI volumes."""

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,        # accepted, unused (kept for API parity)
        local_crops_size=96,          # accepted, unused (kept for API parity)
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

        logger.info("###################################")
        logger.info("Using fMRI multi-crop parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info("###################################")

    @staticmethod
    def _random_resized_crop(scan: torch.Tensor, scale) -> torch.Tensor:
        T, C, X, Y, Z = scan.shape
        frac = float(np.random.uniform(*scale))
        cx, cy, cz = max(1, int(X * frac)), max(1, int(Y * frac)), max(1, int(Z * frac))
        sx = int(np.random.randint(0, max(X - cx, 1)))
        sy = int(np.random.randint(0, max(Y - cy, 1)))
        sz = int(np.random.randint(0, max(Z - cz, 1)))
        sub = scan[:, :, sx:sx + cx, sy:sy + cy, sz:sz + cz]
        return F.interpolate(sub, size=(X, Y, Z),
                             mode="trilinear", align_corners=False)

    def __call__(self, scan: torch.Tensor) -> dict:
        global_crops = [
            self._random_resized_crop(scan, self.global_crops_scale)
            for _ in range(2)
        ]
        local_crops = [
            self._random_resized_crop(scan, self.local_crops_scale)
            for _ in range(self.local_crops_number)
        ]
        return {
            "global_crops":         global_crops,
            "global_crops_teacher": global_crops,
            "local_crops":          local_crops,
            "offsets":              (),
        }
