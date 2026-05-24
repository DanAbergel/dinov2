# Factorised 3D-spatial + 1D-temporal patch embedding for fMRI.
#
# Architecture follows MovieGen TAE TemporalEncoder
# (github.com/MathieuTuli/MovieGen, tae.py:740) but adapted to 4D fMRI:
#   - Conv3Plus1d is the fMRI equivalent of MovieGen Conv2Plus1d:
#     a separable spatial-3D + temporal-1D conv used as the basic block
#     at every level of the encoder (NOT just split between two phases).
#   - ResBlocks contain TWO Conv3Plus1d each (same as TemporalResnetBlock).
#   - Downsamples are Conv3Plus1d with strided spatial AND temporal.
#   - Each hierarchical level uses Conv3Plus1d-based blocks; we never have
#     a "pure spatial then pure temporal" phase.
#
# Strides for fMRI (HCP):
#   Level 0 -> Level 1: spatial stride 3, temporal stride 2  (1200 -> 600, 45 -> 15)
#   Level 1 -> Level 2: spatial stride 3, temporal stride 10 (600 -> 60, 15 -> 5)
# Total: 9x spatial (matches patch_size=9), 20x temporal (= temporal_kernel).
#
# Carries the factorised positional embedding (pos_temporal + pos_spatial
# + pos_cls) added by the ViT's 6D branch in `prepare_tokens_with_masks`.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_


class Conv3Plus1d(nn.Module):
    """Spatial 3D conv + temporal 1D conv applied sequentially.

    fMRI equivalent of MovieGen's `Conv2Plus1d` (tae.py L573):
      - the spatial part is a Conv3d acting per-frame (treats T as batch)
      - the temporal part is a Conv1d acting per-voxel-position (treats
        the spatial grid as batch)

    Together they implement a SEPARABLE 4D conv (T, X, Y, Z) — equivalent
    to a full (T, X, Y, Z) Conv4d but with far fewer parameters and using
    only standard PyTorch primitives (no Conv4d in PyTorch).

    Input/output layout: (B, C, T, X, Y, Z).
    """

    def __init__(
        self, in_c, out_c,
        K_s: int = 3, S_s: int = 1, P_s: int = 1,
        K_t: int = 3, S_t: int = 1, P_t: int = 1,
    ):
        super().__init__()
        self.spatial = nn.Conv3d(in_c, out_c, kernel_size=K_s, stride=S_s, padding=P_s)
        self.temporal = nn.Conv1d(out_c, out_c, kernel_size=K_t, stride=S_t, padding=P_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, X, Y, Z)
        B, _, T, _, _, _ = x.shape
        # Spatial: process each frame independently
        x = rearrange(x, 'b c t x y z -> (b t) c x y z')
        x = self.spatial(x)                                       # (B*T, C', X', Y', Z')
        _, _, X2, Y2, Z2 = x.shape
        # Temporal: process each spatial position independently
        x = rearrange(x, '(b t) c x y z -> (b x y z) c t', b=B, t=T)
        x = self.temporal(x)                                      # (B*X2*Y2*Z2, C', T')
        # Back to 6D
        x = rearrange(x, '(b x y z) c t -> b c t x y z', b=B, x=X2, y=Y2, z=Z2)
        return x


class _ResBlock3Plus1d(nn.Module):
    """Residual block with two Conv3Plus1d. Equivalent to MovieGen's
    `TemporalResnetBlock` (tae.py L643)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = Conv3Plus1d(ch, ch)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = Conv3Plus1d(ch, ch)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class PatchEmbed3DPlus1D(nn.Module):
    """Hierarchical spatio-temporal encoder for fMRI, MovieGen-TAE style.

    Constructor signature is the dinov2 PatchEmbed contract so this drops
    in as the `embed_layer=` of DinoVisionTransformer; the fMRI-specific
    `temporal_size` and `temporal_kernel` are bound via `functools.partial`
    in `build_model_from_cfg`.
    """

    def __init__(
        self,
        img_size,                # (X, Y, Z) — read from data at startup
        temporal_size: int,      # T
        patch_size: int = 9,
        in_chans: int = 1,
        embed_dim: int = 384,
        temporal_kernel: int = 20,
    ) -> None:
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.temporal_size = temporal_size
        self.temporal_kernel = temporal_kernel

        # ----- 3 hierarchical levels with Conv3Plus1d throughout ----------
        # Strides per level transition (spatial, temporal):
        #   conv_in -> level_0: (3, 1)       first spatial downsample baked in
        #   level_0 -> level_1: (3, 2)       9x spatial so far, 2x temporal
        #   level_1 -> level_2: (1, 10)      9x spatial, 20x temporal (= temporal_kernel)
        # Channels: 1 -> 32 -> 64 -> embed_dim.
        #
        # Why downsample at conv_in: with 19200 frames (8 locals x batch 2 x T=1200)
        # going through a full-resolution spatial conv at 16 ch, the intermediate
        # activation reaches 67 GB just before the rearrange, which then needs to
        # COPY the whole tensor to a new layout (another 67 GB) -> OOM. Moving
        # the first spatial downsample into conv_in cuts the intermediate by ~9x.
        # MovieGen TAE can afford to start at full resolution because they have
        # very few frames per video (T=8); we have 1200 frames.

        # Initial projection AND first spatial downsample.
        self.conv_in = Conv3Plus1d(in_chans, 32, K_s=3, S_s=3, P_s=0, K_t=3, S_t=1, P_t=1)

        # Level 0 (at /3 spatial, full temporal; 32 ch).
        self.block_0 = _ResBlock3Plus1d(32)
        # Downsample to /9 spatial total, /2 temporal; 32 -> 64 ch.
        self.down_0 = Conv3Plus1d(32, 64,
                                  K_s=3, S_s=3, P_s=0,
                                  K_t=3, S_t=2, P_t=1)

        # Level 1 (at /9 spatial, /2 temporal; 64 ch).
        self.block_1 = _ResBlock3Plus1d(64)
        # Spatial-stride-1 (already at target) + temporal stride 10; 64 -> embed_dim.
        self.down_1 = Conv3Plus1d(64, embed_dim,
                                  K_s=3, S_s=1, P_s=1,
                                  K_t=10, S_t=10, P_t=0)

        # Level 2 (target resolution = token grid: (T_eff, gx, gy, gz)).
        self.block_2 = _ResBlock3Plus1d(embed_dim)

        # Token-grid sizes.
        gx, gy, gz = (s // patch_size for s in self.img_size)
        self.num_spatial_patches = gx * gy * gz
        self.num_temporal_patches = temporal_size // temporal_kernel
        self.num_patches = self.num_temporal_patches * self.num_spatial_patches

        # Factorised positional embeddings.
        self.pos_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patches, embed_dim))
        self.pos_spatial  = nn.Parameter(torch.zeros(1, self.num_spatial_patches, embed_dim))
        self.pos_cls      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_temporal, std=0.02)
        trunc_normal_(self.pos_spatial,  std=0.02)
        trunc_normal_(self.pos_cls,      std=0.02)

    def combined_patch_pos(self) -> torch.Tensor:
        """(1, T_eff * N_spatial, embed_dim) — broadcast sum of the two
        factorised embeddings. Order: (t outer, n inner)."""
        pos_t = repeat(self.pos_temporal, '1 t d -> 1 (t n) d', n=self.num_spatial_patches)
        pos_s = repeat(self.pos_spatial,  '1 n d -> 1 (t n) d', t=self.num_temporal_patches)
        return pos_t + pos_s

    def _encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single pass through the hierarchical encoder. Wrapped by checkpoint
        in `forward` when in training mode."""
        x = self.conv_in(x)
        x = self.block_0(x)
        x = self.down_0(x)
        x = self.block_1(x)
        x = self.down_1(x)
        x = self.block_2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in: (B, T, in_chans, X, Y, Z)  — dinov2's 6D layout
        if x.ndim != 6:
            raise ValueError(
                f"PatchEmbed3DPlus1D expects shape (B, T, C, X, Y, Z); got {tuple(x.shape)}"
            )

        # Move to (B, C, T, X, Y, Z) layout — Conv3Plus1d expects this.
        x = rearrange(x, 'b t c x y z -> b c t x y z')

        # Activation checkpointing: spatial encoder at full resolution
        # produces large intermediate activations. We trade ~30% extra
        # forward compute (recompute during backward) for ~80 GB memory.
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._encoder_forward, x, use_reentrant=False,
            )
        else:
            x = self._encoder_forward(x)

        # x: (B, embed_dim, T_eff, gx, gy, gz). Rearrange to tokens.
        x = rearrange(x, 'b c t x y z -> b (t x y z) c')          # (B, T_eff*N_spatial, embed_dim)
        return x
