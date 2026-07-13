# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# =============================================================================
# FMRI PROJECT CHANGES (upstream DINOv2 file, modified for our fMRI pipeline)
#   + fmri transform imports    (L38 inline): pull in RandomTokenMaskingGenerator
#     + MaskingAugmentation3D for the fMRI augmentation/masking branches.
#   + int() cast of *_iters      (L94 inline): fractional warmup_epochs break
#     np.linspace; cast scheduler iter counts to int.
#   + apply_freeze_policy        (L151-205, framed): partial-freeze ablation of the
#     student backbone (fmri_only / fmri_plus_last_3); default = official no-freeze.
#   + optimizer_step_and_ema     (L208-235, framed): official step body (clip/step/
#     EMA) extracted so the grad-accum guard in do_train stays readable.
#   + freeze call in do_train    (L254 inline): invoke apply_freeze_policy pre-optim.
#   + fmri token-grid & masking  (L286-315, framed): (T_eff, N_spatial) token count
#     and MAE/BeiT mask generator for PatchEmbed3DPlus1D; else = official 2D path.
#   + fmri augmentation branch   (L325-335, framed): MaskingAugmentation3D transform.
#   + proportional sampler       (L361-376, framed): fixed per-cohort batch quota over
#     MixedFMRIDataset (SamplerType.PROPORTIONAL); else = official sharded-infinite.
#     Plus proportional_quota forwarded to make_data_loader (L387 inline).
#   + grad-accumulation step/EMA (L425-440, framed): accumulate N micro-steps, scale
#     loss by N, step+EMA once per cycle (default N=1 = official single step).
#   Everything else in this file is unchanged upstream DINOv2.
# =============================================================================

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
# FMRI: RandomTokenMaskingGenerator + MaskingAugmentation3D added alongside the
# official transforms for the fmri_augmentation / fmri_masking_only branches below.
from dinov2.data import (
    collate_data_and_cast, DataAugmentationDINO, CellAugmentationDINO,
    MaskingGenerator, RandomTokenMaskingGenerator, MaskingAugmentation3D,
)
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    # FMRI: wrap the four *_iters below in int(). Fractional warmup_epochs
    # (e.g. 0.3) yield floats that break np.linspace (needs an int `num`).
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=int(cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH),
        warmup_iters=int(cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH),
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=int(cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH),
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=int(cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH),
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=int(cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH),
        warmup_iters=int(cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH),
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


# ┌───────────────────────────────────────────────────────────────────────────┐
# │ FMRI ADDITION — not in upstream DINOv2.                                     │
# │ apply_freeze_policy: partial-freeze ablation of the student backbone         │
# │ (fmri_only / fmri_plus_last_3); default no-freeze keeps official behavior.  │
# └───────────────────────────────────────────────────────────────────────────┘
def apply_freeze_policy(model, freeze_mode):
    """Freeze parts of the student backbone before training.

    FMRI CHANGE: ablation requested by Ariel/Yoni. Lets us train only the
    fMRI-specific parts (patch_embed.* = Conv3Plus1d stack + PositionEmbedding3D),
    or also the last 3 transformer blocks + final norm. Default (None / 'none')
    keeps the official behavior: everything is trained.

    Modes:
      None / 'none'         -> no freeze (= official behavior)
      'fmri_only'           -> only patch_embed.* is trainable in the backbone
                               (the rest of the backbone — blocks, cls_token,
                               register_tokens, pos_embed, norm — is frozen).
                               Heads (DINOHead/iBOTHead) stay trainable (random init).
      'fmri_plus_last_3'    -> patch_embed.* + blocks.9-11 + norm. trainable.
                               blocks.0-8 + cls_token + register_tokens + pos_embed frozen.

    Only the BACKBONE is frozen here. The DINO/iBOT heads are random init and
    MUST stay trainable, otherwise the SSL loss is meaningless.
    """
    if freeze_mode in (None, "none", ""):
        return
    backbone = model.student.backbone
    n_frozen = n_train = 0
    for name, p in backbone.named_parameters():
        # FSDP-wrapped names contain '_fsdp_wrapped_module.' segments; strip them
        # so our prefix matching is on the LOGICAL module path.
        logical = name.replace("_fsdp_wrapped_module.", "")
        if freeze_mode == "fmri_only":
            trainable = logical.startswith("patch_embed.")
        elif freeze_mode == "fmri_plus_last_3":
            trainable = (
                logical.startswith("patch_embed.")
                or logical.startswith("blocks.9.")
                or logical.startswith("blocks.10.")
                or logical.startswith("blocks.11.")
                or logical.startswith("norm.")
            )
        else:
            raise ValueError(f"Unknown freeze_pretrained mode: {freeze_mode!r}")
        p.requires_grad_(trainable)
        if trainable:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    logger.info(
        f"FMRI freeze_pretrained={freeze_mode!r}: backbone "
        f"trainable={n_train:,} params, frozen={n_frozen:,} params"
    )
# └── end FMRI ADDITION: apply_freeze_policy ──────────────────────────────────┘


# ┌───────────────────────────────────────────────────────────────────────────┐
# │ FMRI ADDITION — not in upstream DINOv2.                                     │
# │ optimizer_step_and_ema: the official step body (unscale/clip/step/EMA)       │
# │ extracted into one function so the grad-accum guard in do_train stays clean.│
# └───────────────────────────────────────────────────────────────────────────┘
def optimizer_step_and_ema(model, optimizer, fp16_scaler, clip_grad, mom):
    """The official single optimizer step body, extracted into one named
    function: (unscale fp16 grads,) clip, step, update scaler, EMA teacher.

    FMRI CHANGE: this used to be inline in do_train. Pulling it out keeps the
    gradient-accumulation guard in do_train a clean two-liner (zero at the
    start of an N-cycle, this step at the end). The logic itself is unchanged
    from upstream.
    """
    if fp16_scaler is not None:
        if clip_grad:
            fp16_scaler.unscale_(optimizer)
            for v in model.student.values():
                v.clip_grad_norm_(clip_grad)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
    else:
        if clip_grad:
            for v in model.student.values():
                v.clip_grad_norm_(clip_grad)
        optimizer.step()
    model.update_teacher(mom)
# └── end FMRI ADDITION: optimizer_step_and_ema ───────────────────────────────┘


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    # FMRI: optional partial-freeze of the student backbone before the optimizer
    # is built. Default (no YAML flag) = no freeze = official behavior. See apply_freeze_policy.
    apply_freeze_policy(model, getattr(cfg.optim, "freeze_pretrained", None))
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    # ┌───────────────────────────────────────────────────────────────────────────┐
    # │ FMRI ADDITION — not in upstream DINOv2.                                     │
    # │ Compute (n_tokens, mask_generator) for a (T_eff, N_spatial) token grid from │
    # │ PatchEmbed3DPlus1D, where the official scalar img_size does not apply.       │
    # │ fmri_masking_only picks per-token (MAE) masking over BeiT-block masking,    │
    # │ since the flattened order breaks 3D neighbourhoods (meeting 2026-06-14, §2).│
    # │ The else-branch is the unchanged official 2D (img/p, img/p) computation.    │
    # └───────────────────────────────────────────────────────────────────────────┘
    if getattr(cfg.train, "fmri_augmentation", False):
        patch_size = cfg.student.patch_size
        gx, gy, gz = (s // patch_size for s in cfg.student.fmri_img_size)
        n_spatial = gx * gy * gz
        t_eff = cfg.student.fmri_temporal_size // cfg.student.fmri_temporal_kernel
        n_tokens = t_eff * n_spatial
        if getattr(cfg.train, "fmri_masking_only", False):
            mask_generator = RandomTokenMaskingGenerator(input_size=(t_eff, n_spatial))
        else:
            mask_generator = MaskingGenerator(
                input_size=(t_eff, n_spatial),
                max_num_patches=int(0.5 * n_tokens),
            )
    else:
        img_size = cfg.crops.global_crops_size
        patch_size = cfg.student.patch_size
        n_tokens = (img_size // patch_size) ** 2
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
    # └── end FMRI ADDITION: fmri token-grid & mask generator ─────────────────────┘

    if cfg.train.cell_augmentation:
        data_transform = CellAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )
    # ┌───────────────────────────────────────────────────────────────────────────┐
    # │ FMRI ADDITION — not in upstream DINOv2.                                     │
    # │ Third augmentation branch (symmetric to cell_augmentation): masking-only    │
    # │ full-image crops for fMRI; per-token masking is applied later in collate    │
    # │ (meeting 2026-06-14, §2). Selected by cfg.train.fmri_augmentation.          │
    # └───────────────────────────────────────────────────────────────────────────┘
    elif getattr(cfg.train, "fmri_augmentation", False):
        # local_crops_number only: masking-only ignores scale/size; global=2 is a
        # constant inside the class (GLOBAL_CROPS_NUMBER).
        data_transform = MaskingAugmentation3D(cfg.crops.local_crops_number)
    # └── end FMRI ADDITION: fmri masking augmentation branch ─────────────────────┘
    else:
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # ┌───────────────────────────────────────────────────────────────────────────┐
    # │ FMRI ADDITION — not in upstream DINOv2.                                     │
    # │ Per-batch dataset proportions over MixedFMRIDataset: with 5 heterogeneous   │
    # │ cohorts each batch gets a fixed share (default HCP4/ABIDE4/OASIS4/ADNI3/     │
    # │ AOMIC1=16). Enabled by cfg.train.proportional_sampler; optional             │
    # │ proportional_quota overrides. The else-branch is the official sampler.      │
    # └───────────────────────────────────────────────────────────────────────────┘
    proportional_quota = None
    if getattr(cfg.train, "proportional_sampler", False) and hasattr(dataset, "dataset_indices"):
        sampler_type = SamplerType.PROPORTIONAL
        proportional_quota = {str(k): int(v) for k, v in cfg.train.proportional_quota.items()} \
            if cfg.train.get("proportional_quota", None) else None
        logger.info(f"FMRI proportional sampler: quota={proportional_quota or 'default'}")
    else:
        sampler_type = SamplerType.SHARDED_INFINITE  # official DINOv2 default
    # └── end FMRI ADDITION: proportional sampler selection ───────────────────────┘
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
        proportional_quota=proportional_quota,  # FMRI: forwarded to PROPORTIONAL sampler; None => official path
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # FMRI: gradient accumulation over N=grad_accum_steps micro-steps
        # (default 1 = official single-step); consumed by the guard below.
        grad_accum_steps = int(cfg.optim.get("grad_accum_steps", 1))

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        # ┌───────────────────────────────────────────────────────────────────────────┐
        # │ FMRI ADDITION — not in upstream DINOv2.                                     │
        # │ Grad-accumulation guard: zero at the start of each N-cycle, scale the loss  │
        # │ by N inside forward_backward, then step+EMA once at the cycle end.          │
        # │ Upstream did zero_grad / forward_backward / step / EMA every iteration.     │
        # └───────────────────────────────────────────────────────────────────────────┘
        if iteration % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(
            data, teacher_temp=teacher_temp, loss_scale=float(grad_accum_steps),
        )
        if (iteration + 1) % grad_accum_steps == 0:
            optimizer_step_and_ema(
                model, optimizer, fp16_scaler, cfg.optim.clip_grad, mom,
            )
        # └── end FMRI ADDITION: grad-accumulation step/EMA ───────────────────────────┘

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
