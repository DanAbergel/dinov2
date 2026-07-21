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
from .image_folder import ImageFolder  # added for the Imagenette image control test
from .subject_slices import SubjectSliceFolder  # slices grouped by subject (positive pair = 2 slices of same brain)
