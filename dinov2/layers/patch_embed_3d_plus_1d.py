# Factorised 3D-spatial + 1D-temporal patch embedding for fMRI.
#
# Drops in as the `embed_layer=` argument of DinoVisionTransformer
# (constructor signature matches dinov2.layers.PatchEmbed).
#
# Input:  (B, T, in_chans=1, X, Y, Z)         a batch of full fMRI scans
# Output: (B, T_eff * N_spatial, embed_dim)   token sequence ready for the ViT
#
# Carries the factorised positional embedding (pos_temporal, pos_spatial,
# pos_cls) as `nn.Parameter` attributes so the ViT can add them from its
# 6D branch in `prepare_tokens_with_masks` (replaces the official flat
# `self.pos_embed` add). Order matches `src/dino/models.py:287-289 +
# 328-340` of the previous FAIR project:
#   - patch at (t, n) gets pos_temporal[t] + pos_spatial[n]
#   - CLS gets pos_cls
#   - register tokens get no positional embedding (same as official)

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_


class PatchEmbed3DPlus1D(nn.Module):
    """Spatial Conv3d (per-frame) + temporal Conv1d (over frames) + factorised pos."""

    def __init__(
        self,
        img_size,                # (X, Y, Z) — read from the data tensor at startup
        temporal_size: int,      # T — read from the data tensor at startup
        patch_size: int = 9,
        in_chans: int = 1,
        embed_dim: int = 384,
        temporal_kernel: int = 10,
    ) -> None:
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.temporal_size = temporal_size
        self.temporal_kernel = temporal_kernel

        self.spatial = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.temporal = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=temporal_kernel, stride=temporal_kernel,
        )

        # Token-grid sizes (non-overlapping convs: output length = input // kernel).
        gx, gy, gz = (s // patch_size for s in self.img_size)
        self.num_spatial_patches = gx * gy * gz
        self.num_temporal_patches = temporal_size // temporal_kernel
        self.num_patches = self.num_temporal_patches * self.num_spatial_patches

        # Factorised positional embeddings, mirroring FAIR/src/dino/models.py:287-289.
        # Total params: (T_eff + N_spatial + 1) * embed_dim. For T=1200/k=10 and
        # img=45x54x45/p=9: (120 + 150 + 1) * 384 = 104 064 params, vs the flat
        # (1, T_eff*N_spatial+1, embed_dim) = 18001 * 384 = 6 912 384 params.
        self.pos_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patches, embed_dim))
        self.pos_spatial  = nn.Parameter(torch.zeros(1, self.num_spatial_patches, embed_dim))
        self.pos_cls      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_temporal, std=0.02)
        trunc_normal_(self.pos_spatial,  std=0.02)
        trunc_normal_(self.pos_cls,      std=0.02)

    def combined_patch_pos(self) -> torch.Tensor:
        """(1, T_eff * N_spatial, embed_dim) — broadcast sum of the two
        factorised embeddings. Sequence order is (t outer, n inner), matching
        the rearrange '(b n) d t -> b (t n) d' in `forward` below and the
        layout used in FAIR/src/dino/models.py:328-332."""
        pos_t = repeat(self.pos_temporal, '1 t d -> 1 (t n) d', n=self.num_spatial_patches)
        pos_s = repeat(self.pos_spatial,  '1 n d -> 1 (t n) d', t=self.num_temporal_patches)
        return pos_t + pos_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_chans, X, Y, Z)
        if x.ndim != 6:
            raise ValueError(
                f"PatchEmbed3DPlus1D expects shape (B, T, C, X, Y, Z); got {tuple(x.shape)}"
            )
        B = x.shape[0]

        # Step 1: spatial Conv3d applied per-frame (frames flattened into batch).
        x = rearrange(x, 'b t c x y z -> (b t) c x y z')
        x = self.spatial(x)                                       # (B*T, D, gx, gy, gz)
        x = rearrange(
            x, '(b t) d gx gy gz -> (b gx gy gz) d t', b=B,
        )                                                         # (B*N_spatial, D, T)

        # Step 2: temporal Conv1d applied per-spatial-location.
        x = self.temporal(x)                                      # (B*N_spatial, D, T_eff)
        x = rearrange(
            x, '(b n) d t -> b (t n) d', b=B,
        )                                                         # (B, T_eff*N_spatial, D)
        # Positional embedding is added by the ViT's 6D branch (so iBOT
        # masks can replace patches *before* pos is added, matching the
        # official 4D path's ordering).
        return x
