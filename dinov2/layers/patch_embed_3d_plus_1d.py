# Factorised 3D-spatial + 1D-temporal patch embedding for fMRI.
#
# Architecture follows MovieGen TAE TemporalEncoder
# (github.com/MathieuTuli/MovieGen, tae.py:740) but adapted to 4D fMRI:
#   - Conv3Plus1d is the fMRI equivalent of MovieGen Conv2Plus1d:
#     a separable spatial-3D + temporal-1D conv used as the basic block
#     at every level of the encoder (NOT just split between two phases).
#   - ResBlocks contain TWO Conv3Plus1d each (same as TemporalResnetBlock).
#   - Downsampling is done by AvgPool (spatial AND temporal), NOT strided
#     conv: every conv is stride-1 and a following AvgPool does the reduction.
#
# Reductions for the live config (T=270, temporal_kernel=10, img 45x54x45,
# patch_size=9):
#   conv_in:  spatial /3               (45 -> 15)
#   down_0:   spatial /3, temporal /2  (15 -> 5,  270 -> 135)
#   down_1:   temporal /5              (135 -> 27)
# Total: 9x spatial (= patch_size), 10x temporal (= temporal_kernel).
# Token grid: 27 (temporal) x 150 (5*6*5 spatial) = 4050 tokens per crop.
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


class PositionEmbedding3D(nn.Module):
    """Factorised (learned) positional embedding for the fMRI token grid.

    Instead of one flat table of size (T_eff * N_spatial) like a 2D ViT, we
    keep two small learned tables that are broadcast and summed:
        pos = pos_temporal (broadcast over space) + pos_spatial (broadcast over time)
    plus a separate pos_cls for the CLS token. This is O(T_eff + N_spatial)
    parameters instead of O(T_eff * N_spatial).

    Kept as its own nn.Module so positional encoding is a separate concern
    from the patchify conv stack (PatchEmbed3DPlus1D holds one of these).
    """

    def __init__(self, num_temporal_patches: int, num_spatial_patches: int, embed_dim: int):
        super().__init__()
        self.num_temporal_patches = num_temporal_patches
        self.num_spatial_patches = num_spatial_patches
        self.pos_temporal = nn.Parameter(torch.zeros(1, num_temporal_patches, embed_dim))
        self.pos_spatial  = nn.Parameter(torch.zeros(1, num_spatial_patches, embed_dim))
        self.pos_cls      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_temporal, std=0.02)
        trunc_normal_(self.pos_spatial,  std=0.02)
        trunc_normal_(self.pos_cls,      std=0.02)

    def combined_patch_pos(self, t_eff: int = None) -> torch.Tensor:
        """(1, t_eff * N_spatial, embed_dim) — broadcast sum of the two factorised
        embeddings. Order: (t outer, n inner).

        `t_eff` defaults to the full `num_temporal_patches`. Passing a smaller value
        (a local crop that is a temporal sub-window) slices the temporal table to its
        first `t_eff` positions — the temporal analogue of DINOv2's spatial pos-embed
        interpolation for smaller local crops. Full-volume crops pass t_eff=None and
        behave exactly as before.
        """
        n_t = self.num_temporal_patches if t_eff is None else int(t_eff)
        pos_t = repeat(self.pos_temporal[:, :n_t], '1 t d -> 1 (t n) d', n=self.num_spatial_patches)
        pos_s = repeat(self.pos_spatial,  '1 n d -> 1 (t n) d', t=n_t)
        return pos_t + pos_s


class PatchEmbed3DPlus1D(nn.Module):
    """Hierarchical spatio-temporal encoder for fMRI, MovieGen-TAE style.

    Constructor signature is the dinov2 PatchEmbed contract so this drops
    in as the `embed_layer=` of DinoVisionTransformer; the fMRI-specific
    `temporal_size` and `temporal_kernel` are bound via `functools.partial`
    in `build_model_from_cfg`.

    Downsampling is done by AvgPool on stride-1 conv outputs (spatial and
    temporal), so the token grid is a pure function of img_size / patch_size
    and temporal_size / temporal_kernel.
    """

    def __init__(
        self,
        img_size,                # (X, Y, Z) — read from data at startup
        temporal_size: int,      # T
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

        # ----- 3 hierarchical levels with Conv3Plus1d throughout ----------
        # Every conv is stride-1; each level's reduction is a following AvgPool
        # (spatial and/or temporal). Channels: 1 -> 32 -> 64 -> embed_dim.
        #
        # Reductions (spatial, temporal) for T=270, temporal_kernel=10:
        #   conv_in -> level_0: (/3, /1)   first spatial /3            (45 -> 15)
        #   level_0 -> level_1: (/3, /2)   9x spatial, 2x temporal     (15 -> 5, 270 -> 135)
        #   level_1 -> level_2: (/1, /5)   9x spatial, 10x temporal    (135 -> 27)
        # -> token grid 27 x 150 = 4050 per crop.

        # Initial projection, stride-1 (spatial /3 by AvgPool in forward).
        self.conv_in = Conv3Plus1d(in_chans, 32, K_s=3, S_s=1, P_s=1, K_t=3, S_t=1, P_t=1)

        # Level 0 (at /3 spatial, full temporal; 32 ch).
        self.block_0 = _ResBlock3Plus1d(32)
        # Stride-1; spatial /3 and temporal /2 by AvgPool in forward.
        self.down_0 = Conv3Plus1d(32, 64, K_s=3, S_s=1, P_s=1, K_t=3, S_t=1, P_t=1)

        # Level 1 (at /9 spatial, /2 temporal; 64 ch).
        self.block_1 = _ResBlock3Plus1d(64)
        # Total temporal reduction = conv_in(1) * down_0(2) * down_1(down_1_kt)
        # must equal temporal_kernel, so down_1_kt = temporal_kernel // 2. Deriving
        # it from temporal_kernel (not hardcoding) makes the whole architecture a
        # function of the config, so a checkpoint trained with temporal_kernel=20
        # and one with 10 are both rebuildable from their saved config.yaml.
        down_1_kt = temporal_kernel // 2
        # Stride-1; temporal /down_1_kt by AvgPool in forward.
        self.down_1 = Conv3Plus1d(64, embed_dim, K_s=3, S_s=1, P_s=1, K_t=3, S_t=1, P_t=1)
        # AvgPool factors. Spatial: /3 after conv_in and /3 after down_0 (patch_size
        # 9 = 3x3). Temporal: /2 after down_0, /down_1_kt after down_1 (product =
        # temporal_kernel).
        self.spool_k = 3
        self.pool0_k, self.pool1_k = 2, down_1_kt

        # Level 2 (target resolution = token grid: (T_eff, gx, gy, gz)).
        self.block_2 = _ResBlock3Plus1d(embed_dim)

        # Token-grid sizes. num_patches is read by the ViT __init__ to size
        # its (unused-in-fMRI) flat pos_embed, so we expose it here.
        gx, gy, gz = (s // patch_size for s in self.img_size)
        self.num_spatial_patches = gx * gy * gz
        self.num_temporal_patches = temporal_size // temporal_kernel
        self.num_patches = self.num_temporal_patches * self.num_spatial_patches

        # Factorised (learned) positional embedding, a separate nn.Module.
        self.pos = PositionEmbedding3D(
            self.num_temporal_patches, self.num_spatial_patches, embed_dim)

    @staticmethod
    def _tpool(x: torch.Tensor, k: int) -> torch.Tensor:
        """Average-pool the temporal axis (dim 2) of (B,C,T,X,Y,Z) by factor k."""
        B, C, T, X, Y, Z = x.shape
        x = rearrange(x, 'b c t x y z -> (b c x y z) t').unsqueeze(1)   # (N,1,T)
        x = F.avg_pool1d(x, kernel_size=k, stride=k).squeeze(1)         # (N, T//k)
        return rearrange(x, '(b c x y z) t -> b c t x y z', b=B, c=C, x=X, y=Y, z=Z)

    @staticmethod
    def _spool(x: torch.Tensor, k: int) -> torch.Tensor:
        """Average-pool the 3 spatial axes (X,Y,Z) of (B,C,T,X,Y,Z) by factor k."""
        B, C, T, X, Y, Z = x.shape
        x = rearrange(x, 'b c t x y z -> (b c t) x y z').unsqueeze(1)   # (N,1,X,Y,Z)
        x = F.avg_pool3d(x, kernel_size=k, stride=k).squeeze(1)         # (N, X/k, Y/k, Z/k)
        return rearrange(x, '(b c t) x y z -> b c t x y z', b=B, c=C, t=T)

    def _encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single pass through the hierarchical encoder. Wrapped by checkpoint
        in `forward` when training. Every downsample is an AvgPool on a stride-1
        conv output: spatial after conv_in and down_0, temporal after down_0 and
        down_1."""
        x = self.conv_in(x)
        x = self._spool(x, self.spool_k)              # spatial /3
        x = self.block_0(x)
        x = self.down_0(x)
        x = self._spool(x, self.spool_k)              # spatial /3
        x = self._tpool(x, self.pool0_k)              # temporal /2
        x = self.block_1(x)
        x = self.down_1(x)
        x = self._tpool(x, self.pool1_k)              # temporal /down_1_kt
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
