# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# =============================================================================
# FMRI PROJECT CHANGES (upstream DINOv2 file, modified for our fMRI pipeline)
#   + embed_layer plumbing in build_model  (L25, L43-57, framed below):
#     forward an optional `embed_layer` through vit_kwargs so the ViT can use a
#     custom patch embedding (PatchEmbed3DPlus1D) without forking the ViT factory.
#   + fMRI branch in build_model_from_cfg  (L71-97, framed below): when
#     cfg.student.fmri_mode is set, build the PatchEmbed3DPlus1D partial and pass
#     it plus the 3D fmri_img_size through, keeping fMRI plumbing in the config.
#   Everything else in this file is unchanged upstream DINOv2.
# =============================================================================

import logging

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224, embed_layer=None):  # FMRI: added `embed_layer` param
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            in_chans=args.in_chans,
            channel_adaptive=args.channel_adaptive,
        )
        # ┌───────────────────────────────────────────────────────────────────────────┐
        # │ FMRI ADDITION — not in upstream DINOv2.                                     │
        # │ Forward an optional custom patch embedding (PatchEmbed3DPlus1D) via         │
        # │ vit_kwargs; upstream always lets the ViT pick its default 2D PatchEmbed.    │
        # └───────────────────────────────────────────────────────────────────────────┘
        # FMRI CHANGE: forward an optional `embed_layer` to the ViT.
        # OFFICIAL: build_model always lets the ViT pick its default
        # `PatchEmbed` (2D Conv2d). WHY: fMRI needs PatchEmbed3DPlus1D,
        # but `vits.__dict__[arch]` is the unmodified factory function.
        # Passing embed_layer through vit_kwargs is the smallest possible
        # change that lets us swap the patch embedding without forking
        # vision_transformer.py.
        if embed_layer is not None:
            vit_kwargs["embed_layer"] = embed_layer
        # └── end FMRI ADDITION: embed_layer plumbing ─────────────────────────────────┘
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    # ┌───────────────────────────────────────────────────────────────────────────┐
    # │ FMRI ADDITION — not in upstream DINOv2.                                     │
    # │ When cfg.student.fmri_mode is set, build a PatchEmbed3DPlus1D partial and   │
    # │ override img_size with the 3D fmri_img_size; upstream body is just the      │
    # │ single `return build_model(...)` call at the bottom of this box.            │
    # └───────────────────────────────────────────────────────────────────────────┘
    # FMRI CHANGE: if cfg.student.fmri_mode is set, build the
    # PatchEmbed3DPlus1D partial here and pass it through, plus override
    # img_size with the 3D tuple. OFFICIAL: always uses
    # cfg.crops.global_crops_size (int). WHY: keeps fMRI plumbing in the
    # config rather than forking SSLMetaArch or `do_train`.
    embed_layer = None
    img_size = cfg.crops.global_crops_size
    if getattr(cfg.student, "fmri_mode", False):
        from functools import partial
        from dinov2.layers.patch_embed_3d_plus_1d import PatchEmbed3DPlus1D
        embed_layer = partial(
            PatchEmbed3DPlus1D,
            temporal_size=cfg.student.fmri_temporal_size,
            temporal_kernel=cfg.student.fmri_temporal_kernel,
        )
        img_size = tuple(cfg.student.fmri_img_size)
    return build_model(
        cfg.student, only_teacher=only_teacher,
        img_size=img_size, embed_layer=embed_layer,
    )
    # └── end FMRI ADDITION: fmri_mode build_model_from_cfg ────────────────────────┘
