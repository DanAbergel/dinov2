# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .adapters import DatasetWithEnumeratedTargets
from .loaders import make_data_loader, make_dataset, SamplerType
from .collate import collate_data_and_cast
from .masking import MaskingGenerator, RandomTokenMaskingGenerator
from .augmentations import DataAugmentationDINO
from .cell_dino.augmentations import CellAugmentationDINO
# FMRI CHANGE: surface MultiCrop3D / MaskingAugmentation3D at the same import
# level as DataAugmentationDINO / CellAugmentationDINO. WHY: `do_train` selects
# the augmentation class via a single import from this package; we add our fMRI
# branches by symmetry, not by sneaking around the package layout.
from .fmri_data import MultiCrop3D, MaskingAugmentation3D
from .accumulators import NoOpAccumulator, ResultsAccumulator
