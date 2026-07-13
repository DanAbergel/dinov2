# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# =============================================================================
# FMRI PROJECT CHANGES (upstream DINOv2 file, modified for our fMRI pipeline)
#   + RandomTokenMaskingGenerator export  (L15): re-export our per-token masking
#     generator alongside upstream MaskingGenerator.
#   + MaskingAugmentation3D import  (L16): surface our 3D augmentation at the same
#     import level as DataAugmentationDINO so `do_train` selects it with one import.
#   Everything else in this file is unchanged upstream DINOv2.
# =============================================================================

from .adapters import DatasetWithEnumeratedTargets
from .loaders import make_data_loader, make_dataset, SamplerType
from .collate import collate_data_and_cast
from .masking import MaskingGenerator, RandomTokenMaskingGenerator  # FMRI: + RandomTokenMaskingGenerator
from .augmentations import DataAugmentationDINO
from .cell_dino.augmentations import CellAugmentationDINO
from .fmri_data import MaskingAugmentation3D  # FMRI: 3D augmentation for do_train
from .accumulators import NoOpAccumulator, ResultsAccumulator
