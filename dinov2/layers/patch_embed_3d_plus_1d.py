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
# Strides for fMRI (Mixed HCP+ADNI at T=140):
#   conv_in: spatial stride 3 (45 -> 15)
#   Level 0 -> Level 1: spatial stride 3, temporal stride 2  (140 -> 70, 15 -> 5)
#   Level 1 -> Level 2: spatial stride 1, temporal stride 7  (70 -> 10)
# Total: 9x spatial (matches patch_size=9), 14x temporal (= temporal_kernel).
#
# Carries the factorised positional embedding (pos_temporal + pos_spatial
# + pos_cls) added by the ViT's 6D branch in `prepare_tokens_with_masks`.

import math

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


class FourierFeatures3D(nn.Module):
    """Fourier-feature map of 3D coordinates (Tancik et al., NeurIPS 2020).

        gamma(v) = [cos(2*pi * B v), sin(2*pi * B v)],  v in R^3, B in R^{F x 3}

    B is sampled once from N(0, sigma^2) and registered as a buffer (fixed,
    NOT learned) for the first iteration. Nearby 3D positions map to similar
    features at low frequencies but become distinguishable at high frequencies.
    """

    def __init__(self, num_freqs: int = 32, sigma: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(num_freqs, 3) * sigma)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:   # (N, 3) -> (N, 2*F)
        proj = 2 * math.pi * positions @ self.B.t()               # (N, F)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PositionEmbedding3D(nn.Module):
    """Factorised positional embedding for the fMRI token grid.

    Instead of one flat table of size (T_eff * N_spatial) like a 2D ViT, we
    keep two small tables that are broadcast and summed:
        pos = pos_temporal (broadcast over space) + pos_spatial (broadcast over time)
    plus a separate pos_cls for the CLS token. This is O(T_eff + N_spatial)
    parameters instead of O(T_eff * N_spatial).

    spatial_mode:
      'learned' (default) -> pos_spatial is a learned table (original behaviour).
      'fourier'           -> pos_spatial is computed from Fourier features of the
                             3D patch-grid coordinates + a small MLP (meeting
                             2026-06-14 §3). pos_temporal and pos_cls stay learned.

    Kept as its own nn.Module so positional encoding is a separate concern
    from the patchify conv stack (PatchEmbed3DPlus1D holds one of these).
    """

    def __init__(self, num_temporal_patches: int, num_spatial_patches: int, embed_dim: int,
                 grid=None, spatial_mode: str = "learned",
                 num_freqs: int = 32, sigma: float = 10.0):
        super().__init__()
        self.num_temporal_patches = num_temporal_patches
        self.num_spatial_patches = num_spatial_patches
        self.spatial_mode = spatial_mode
        self.pos_temporal = nn.Parameter(torch.zeros(1, num_temporal_patches, embed_dim))
        self.pos_cls      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_temporal, std=0.02)
        trunc_normal_(self.pos_cls,      std=0.02)

        if spatial_mode == "learned":
            self.pos_spatial = nn.Parameter(torch.zeros(1, num_spatial_patches, embed_dim))
            trunc_normal_(self.pos_spatial, std=0.02)
        elif spatial_mode == "fourier":
            assert grid is not None, "fourier spatial pos needs the (gx, gy, gz) grid"
            gx, gy, gz = grid
            assert gx * gy * gz == num_spatial_patches, "grid does not match N_spatial"
            self.register_buffer("coords", self._make_grid_coords(gx, gy, gz))
            self.fourier = FourierFeatures3D(num_freqs, sigma)
            self.spatial_proj = nn.Sequential(
                nn.Linear(2 * num_freqs, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            raise ValueError(f"unknown spatial_mode {spatial_mode!r}")

    @staticmethod
    def _make_grid_coords(gx, gy, gz) -> torch.Tensor:
        """(N_spatial, 3) coords in [-1, 1], in the SAME order the patchify
        flattens spatial tokens: 'b c t x y z -> b (t x y z) c' (x outer, z inner)."""
        xs, ys, zs = (torch.linspace(-1, 1, g) for g in (gx, gy, gz))
        gxx, gyy, gzz = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([gxx.flatten(), gyy.flatten(), gzz.flatten()], dim=-1)

    def get_spatial(self) -> torch.Tensor:
        """(1, N_spatial, embed_dim) — learned table or Fourier-derived."""
        if self.spatial_mode == "learned":
            return self.pos_spatial
        # coords/B are float32 buffers; the model runs in fp16, so the cos/sin
        # features are float32 while spatial_proj weights are half -> dtype
        # mismatch. Compute Fourier in float32 (precise), then cast to the
        # Linear's dtype.
        feats = self.fourier(self.coords).to(self.spatial_proj[0].weight.dtype)
        return self.spatial_proj(feats).unsqueeze(0)

    def combined_patch_pos(self) -> torch.Tensor:
        """(1, T_eff * N_spatial, embed_dim) — broadcast sum of the two
        factorised embeddings. Order: (t outer, n inner)."""
        pos_t = repeat(self.pos_temporal, '1 t d -> 1 (t n) d', n=self.num_spatial_patches)
        pos_s = repeat(self.get_spatial(), '1 n d -> 1 (t n) d', t=self.num_temporal_patches)
        return pos_t + pos_s


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
        temporal_kernel: int = 14,
        fourier_pos: bool = False,        # spatial pos: Fourier features vs learned table
        fourier_num_freqs: int = 32,
        fourier_sigma: float = 10.0,
        remove_block2: bool = False,      # point 2: drop the final ResBlock (~83% of params)
        pool_downsample: bool = False,    # point 2: ALL downsampling (spatial AND temporal)
                                          # by stride-1 conv + AvgPool, not strided conv
    ) -> None:
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.temporal_size = temporal_size
        self.temporal_kernel = temporal_kernel
        self.pool_downsample = pool_downsample

        # ----- 3 hierarchical levels with Conv3Plus1d throughout ----------
        # Strides per level transition (spatial, temporal):
        #   conv_in -> level_0: (3, 1)       first spatial downsample baked in
        #   level_0 -> level_1: (3, 2)       9x spatial so far, 2x temporal
        #   level_1 -> level_2: (1, 7)       9x spatial, 14x temporal (= temporal_kernel)
        # Channels: 1 -> 32 -> 64 -> embed_dim.
        #
        # For mixed-dataset training at T=140 (HCP random window + ADNI native):
        #   T=140 -> down_0 -> 70 -> down_1 -> 10 (= T_eff)
        # Token grid: 10 x 150 = 1500 per crop. Much smaller than T=1200 -> 9000.

        # Initial projection + first spatial /3. Strided conv (default) OR stride-1
        # conv keeping full res (+ AvgPool /3 in forward) when pool_downsample.
        self.conv_in = Conv3Plus1d(in_chans, 32,
                                   K_s=3, S_s=(1 if pool_downsample else 3),
                                   P_s=(1 if pool_downsample else 0),
                                   K_t=3, S_t=1, P_t=1)

        # Level 0 (at /3 spatial, full temporal; 32 ch).
        self.block_0 = _ResBlock3Plus1d(32)
        # Downsample: spatial /3 (total /9) and temporal /2. Either strided conv
        # (default) or stride-1 conv + AvgPool (pool_downsample) on BOTH axes.
        self.down_0 = Conv3Plus1d(32, 64,
                                  K_s=3, S_s=(1 if pool_downsample else 3),
                                  P_s=(1 if pool_downsample else 0),
                                  K_t=3, S_t=(1 if pool_downsample else 2), P_t=1)

        # Level 1 (at /9 spatial, /2 temporal; 64 ch).
        self.block_1 = _ResBlock3Plus1d(64)
        # Spatial-stride-1 (already at target) + temporal stride down_1_kt.
        # Total temporal stride = conv_in(1) * down_0(2) * down_1(down_1_kt)
        # = 2 * down_1_kt, which must equal temporal_kernel. So:
        #   down_1_kt = temporal_kernel // 2.
        # Deriving this from temporal_kernel (instead of hardcoding) makes the
        # whole architecture a function of the config, so a checkpoint trained
        # with temporal_kernel=20 (down_1 K_t=10) and one with 14 (K_t=7) are
        # both rebuildable from their saved config.yaml. WHY THIS MATTERS:
        # previously down_1 K_t was hardcoded, so probing an old checkpoint
        # with a different kernel raised a state_dict size mismatch.
        down_1_kt = temporal_kernel // 2
        if pool_downsample:
            # stride-1 conv (preserve T), temporal /down_1_kt done by AvgPool below.
            self.down_1 = Conv3Plus1d(64, embed_dim, K_s=3, S_s=1, P_s=1,
                                      K_t=3, S_t=1, P_t=1)
        else:
            self.down_1 = Conv3Plus1d(64, embed_dim, K_s=3, S_s=1, P_s=1,
                                      K_t=down_1_kt, S_t=down_1_kt, P_t=0)
        # AvgPool factors (only used if pool_downsample). Spatial: /3 after conv_in
        # and /3 after down_0 (patch_size 9 = 3x3). Temporal: /2 after down_0,
        # /down_1_kt after down_1 (product = temporal_kernel). Token grid unchanged.
        self.spool_k = 3
        self.pool0_k, self.pool1_k = 2, down_1_kt

        # Level 2 (target resolution = token grid: (T_eff, gx, gy, gz)).
        # point 2: block_2 is ~83% of the patchify params -> optionally dropped.
        self.block_2 = None if remove_block2 else _ResBlock3Plus1d(embed_dim)

        # Token-grid sizes. num_patches is read by the ViT __init__ to size
        # its (unused-in-fMRI) flat pos_embed, so we expose it here.
        gx, gy, gz = (s // patch_size for s in self.img_size)
        self.num_spatial_patches = gx * gy * gz
        self.num_temporal_patches = temporal_size // temporal_kernel
        self.num_patches = self.num_temporal_patches * self.num_spatial_patches

        # Factorised positional embedding (separate nn.Module). Spatial part is
        # either a learned table or Fourier features of the 3D grid coords.
        self.pos = PositionEmbedding3D(
            self.num_temporal_patches, self.num_spatial_patches, embed_dim,
            grid=(gx, gy, gz),
            spatial_mode="fourier" if fourier_pos else "learned",
            num_freqs=fourier_num_freqs, sigma=fourier_sigma,
        )

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
        in `forward` when in training mode. In pool_downsample mode, every strided
        downsample is done here as AvgPool instead (spatial after conv_in and down_0,
        temporal after down_0 and down_1)."""
        x = self.conv_in(x)
        if self.pool_downsample:
            x = self._spool(x, self.spool_k)              # spatial /3
        x = self.block_0(x)
        x = self.down_0(x)
        if self.pool_downsample:
            x = self._spool(x, self.spool_k)          # spatial /3 by AvgPool
            x = self._tpool(x, self.pool0_k)          # temporal /2 by AvgPool
        x = self.block_1(x)
        x = self.down_1(x)
        if self.pool_downsample:
            x = self._tpool(x, self.pool1_k)          # temporal /down_1_kt by AvgPool
        if self.block_2 is not None:
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
