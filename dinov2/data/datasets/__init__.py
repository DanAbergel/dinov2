# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# =============================================================================
# FMRI PROJECT CHANGES (upstream DINOv2 file, modified for our fMRI pipeline)
#   + MixedFMRIDataset import  (L13): re-export the multi-source dataset so
#     `_parse_dataset_str` (loaders.py) can resolve the "Mixed" dataset string,
#     the same way ImageNet is resolved.
#   Everything else in this file is unchanged upstream DINOv2.
# =============================================================================

from .image_net import ImageNet
from .image_net_22k import ImageNet22k
from .cell_dino.hpaone import HPAone
from .cell_dino.hpafov import HPAFoV
from .cell_dino.chammi_cp import CHAMMI_CP
from .cell_dino.chammi_hpa import CHAMMI_HPA
from .cell_dino.chammi_wtc import CHAMMI_WTC
from ..fmri_data import MixedFMRIDataset  # FMRI: multi-source dataset for _parse_dataset_str
