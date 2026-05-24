# Factorised 3D-spatial + 1D-temporal patch embedding for fMRI.
#
# Hierarchical encoder inspired by MovieGen TAE
# (github.com/MathieuTuli/MovieGen, tae.py:740 TemporalEncoder):
# three spatial stages with stride-3 downsamples + ResBlocks at each
# stage, then two temporal stages with Conv1d + ResBlocks. Token grid:
# T_eff = T // temporal_kernel, N_spatial = prod(img // patch_size).
#
# Carries the factorised positional embedding (pos_temporal +
# pos_spatial + pos_cls) as `nn.Parameter` attributes, added by the
# ViT's 6D branch in `prepare_tokens_with_masks`.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_


class _ResBlock3D(nn.Module):
    """3D residual block: GN + SiLU + Conv3d, twice, with skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _ResBlock1D(nn.Module):
    """1D residual block for the temporal axis."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class PatchEmbed3DPlus1D(nn.Module):
    """Spatial 3D + temporal 1D patchify with factorised pos."""

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

        # ----- Spatial hierarchical encoder -------------------------------
        # Three stages with stride-3 downsamples (total spatial stride: 9).
        # For img (45, 54, 45):
        #   stem      -> (45, 54, 45) @ 16 ch
        #   down 1    -> (15, 18, 15) @ 64 ch
        #   down 2    -> (5, 6, 5)    @ embed_dim
        self.spatial = nn.Sequential(
            # Stem
            nn.Conv3d(in_chans, 16, kernel_size=3, padding=1),
            _ResBlock3D(16),
            # Downsample 1 (stride 3)
            nn.Conv3d(16, 64, kernel_size=3, stride=3),
            _ResBlock3D(64),
            # Downsample 2 (stride 3, to target embed_dim)
            nn.Conv3d(64, embed_dim, kernel_size=3, stride=3),
            _ResBlock3D(embed_dim),
        )
        # ----- Temporal hierarchical encoder ------------------------------
        # Two stages, total temporal stride 10 (= temporal_kernel for fMRI).
        #   down 1: Conv1d stride 2 -> T 1200 -> 600
        #   down 2: Conv1d stride 5 -> T 600  -> 120
        self.temporal = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            _ResBlock1D(embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=5),
            _ResBlock1D(embed_dim),
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_chans, X, Y, Z)
        if x.ndim != 6:
            raise ValueError(
                f"PatchEmbed3DPlus1D expects shape (B, T, C, X, Y, Z); got {tuple(x.shape)}"
            )
        B = x.shape[0]

        # Step 1: spatial encoder applied per-frame.
        x = rearrange(x, 'b t c x y z -> (b t) c x y z')
        x = self.spatial(x)                                       # (B*T, embed_dim, gx, gy, gz)
        x = rearrange(
            x, '(b t) d gx gy gz -> (b gx gy gz) d t', b=B,
        )                                                         # (B*N_spatial, embed_dim, T)

        # Step 2: temporal encoder applied per-spatial-location.
        x = self.temporal(x)                                      # (B*N_spatial, embed_dim, T_eff)
        x = rearrange(
            x, '(b n) d t -> b (t n) d', b=B,
        )                                                         # (B, T_eff*N_spatial, embed_dim)
        return x
