# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .image_net import ImageNet
from .image_net_22k import ImageNet22k
from .cell_dino.hpaone import HPAone
from .cell_dino.hpafov import HPAFoV
from .cell_dino.chammi_cp import CHAMMI_CP
from .cell_dino.chammi_hpa import CHAMMI_HPA
from .cell_dino.chammi_wtc import CHAMMI_WTC
# FMRI CHANGE: re-export HCP / ADNI Dataset classes so `_parse_dataset_str`
# (loaders.py) can resolve "HCP" / "ADNI" dataset strings the same way it
# resolves "ImageNet". WHY: avoid touching loaders.py beyond a couple of
# elif branches; the heavy lifting lives in dinov2/data/fmri_data.py.
from ..fmri_data import HCPFullScanDataset, ADNIFullScanDataset
