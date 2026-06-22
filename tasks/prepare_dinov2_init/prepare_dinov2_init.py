"""Prepare the DINOv2 ImageNet init checkpoint for fMRI transfer learning.

Downloads the official DINOv2 ViT-S/14 with 4 registers ImageNet weights and
strips the keys whose shapes don't match our fMRI model:
  - patch_embed.*  (DINOv2 has a 2D Conv2d patch embed; we use PatchEmbed3DPlus1D)
  - pos_embed      (DINOv2 has a flat 2D pos table; we use factorised 3D pos)

Everything else (the 12 transformer blocks, cls_token, register_tokens, norm)
has identical ViT-S dims (embed_dim=384, depth=12, 4 registers) and loads fine.

Saved as {"model": filtered_state_dict} — the format ssl_meta_arch.py expects:
    student_backbone.load_state_dict(chkpt["model"], strict=False)

So at training start the transformer blocks start from ImageNet DINOv2 while the
3D patchify + pos are random — the "DINOv2 init" transfer-learning setup. This is
the winning config in FREEZE_ABLATION_RESULTS (strategies A full-FT and
C freeze-except-input both build on this init).

Needs internet -> run on the GATEWAY (compute nodes are offline).

Usage:
    python prepare_dinov2_init.py --out /sci/.../checkpoints/dinov2_vits14_reg4_fmri_init.pth
"""

import argparse
from pathlib import Path

import torch

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
DROP_PREFIXES = ("patch_embed.", "pos_embed")   # shape-incompatible with the fMRI model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .pth path")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DINOv2 ViT-S/14 reg4 weights from:\n  {args.url}")
    sd = torch.hub.load_state_dict_from_url(args.url, map_location="cpu")
    # The official file is a flat backbone state_dict.
    if "model" in sd:                       # be robust if it's already wrapped
        sd = sd["model"]
    print(f"  loaded {len(sd)} keys")

    kept, dropped = {}, []
    for k, v in sd.items():
        if k.startswith(DROP_PREFIXES):
            dropped.append(k)
        else:
            kept[k] = v

    print(f"  dropped {len(dropped)} shape-incompatible keys:")
    for k in dropped:
        print(f"    - {k}  {tuple(sd[k].shape)}")
    print(f"  kept {len(kept)} keys (blocks / cls / registers / norm)")

    torch.save({"model": kept}, out)
    print(f"\nSaved init checkpoint -> {out}")
    print("Set student.pretrained_weights to this path in fmri_vits.yaml.")


if __name__ == "__main__":
    main()
