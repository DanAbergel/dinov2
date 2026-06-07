"""Exact trainable-parameter count per ablation strategy.

Builds the student model (backbone + DINOHead + iBOTHead) from each
strategy's saved config.yaml, applies the freeze policy, and prints
trainable / frozen / total counts for backbone, heads, and student total.

Run on Moriah from the FAIR_official root:
    python3 scripts/count_trainable_params.py

The numbers from the .log files (`backbone trainable=...`) only report the
backbone -- this script adds the heads, which are always trainable (random
init at start of training).
"""

import sys
from pathlib import Path
from collections import defaultdict

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov2.configs import dinov2_default_config
from dinov2.train.train import apply_freeze_policy

# We avoid SSLMetaArch (which needs distributed init). Instead, build the
# pieces directly: backbone via build_model_from_cfg, heads via DINOHead.
from dinov2.models import build_model_from_cfg
from dinov2.layers.dino_head import DINOHead


def build_student_pieces(cfg):
    """Return (backbone, dino_head, ibot_head_or_None). Mirrors SSLMetaArch."""
    student_backbone, _, embed_dim = build_model_from_cfg(cfg)
    dino_head = DINOHead(
        in_dim=embed_dim,
        out_dim=cfg.dino.head_n_prototypes,
        hidden_dim=cfg.dino.head_hidden_dim,
        bottleneck_dim=cfg.dino.head_bottleneck_dim,
        nlayers=cfg.dino.head_nlayers,
    )
    ibot_head = None
    if cfg.ibot.separate_head:
        ibot_head = DINOHead(
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
    return student_backbone, dino_head, ibot_head


def fake_student_wrap(backbone, dino_head, ibot_head):
    """A tiny wrapper exposing `.student.backbone` so apply_freeze_policy
    can operate on it without spinning up the full SSLMetaArch."""
    import torch.nn as nn
    class _Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.dino_head = dino_head
            if ibot_head is not None:
                self.ibot_head = ibot_head
    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = _Student()
    return _Wrap()


def count(model, freeze_mode):
    apply_freeze_policy(model, freeze_mode)
    backbone_tr = sum(p.numel() for p in model.student.backbone.parameters() if p.requires_grad)
    backbone_fr = sum(p.numel() for p in model.student.backbone.parameters() if not p.requires_grad)
    head_tr     = sum(p.numel() for p in model.student.dino_head.parameters() if p.requires_grad)
    if hasattr(model.student, "ibot_head"):
        head_tr += sum(p.numel() for p in model.student.ibot_head.parameters() if p.requires_grad)
    return backbone_tr, backbone_fr, head_tr


CONFIGS = [
    ("A (full FT)",     "dinov2/configs/train/fmri_vits_hcp_baseline.yaml",      None),
    ("B (freeze_last3)","dinov2/configs/train/fmri_vits_hcp_freeze_last3.yaml",  "fmri_plus_last_3"),
    ("C (freeze_fmri)", "dinov2/configs/train/fmri_vits_hcp_freeze_fmri.yaml",   "fmri_only"),
]

print(f"{'Strategy':<20} {'Backbone trainable':>22} {'Backbone frozen':>20} {'Heads trainable':>20} {'TOTAL trainable':>20}")
print("-" * 110)
for tag, yaml_path, freeze_mode in CONFIGS:
    cfg = OmegaConf.merge(
        OmegaConf.create(dinov2_default_config),
        OmegaConf.load(yaml_path),
    )
    cfg.optim.freeze_pretrained = freeze_mode  # the freeze YAMLs set this; A's YAML doesn't (None)
    bb, dh, ih = build_student_pieces(cfg)
    model = fake_student_wrap(bb, dh, ih)
    bb_tr, bb_fr, head_tr = count(model, freeze_mode)
    total_tr = bb_tr + head_tr
    print(f"{tag:<20} {bb_tr:>22,} {bb_fr:>20,} {head_tr:>20,} {total_tr:>20,}")
