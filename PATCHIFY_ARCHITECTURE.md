---
title: "PatchEmbed3DPlus1D — Patchify architecture for fMRI"
date: "2026-06-14"
geometry: margin=2cm
fontsize: 10pt
header-includes:
  - \usepackage{tikz}
  - \usetikzlibrary{positioning, arrows.meta, fit, backgrounds, shapes.geometric, calc}
  - \definecolor{cInput}{RGB}{255,224,178}
  - \definecolor{cConv}{RGB}{179,205,255}
  - \definecolor{cRes}{RGB}{195,230,203}
  - \definecolor{cDown}{RGB}{255,179,179}
  - \definecolor{cOut}{RGB}{215,179,255}
  - \definecolor{cShape}{RGB}{120,120,120}
---

# Overview

The patchify module **`PatchEmbed3DPlus1D`** is the fMRI-specific input encoder that turns a 4D scan `(T, X, Y, Z)` into a flat sequence of `(T_eff × N_spatial)` tokens consumed by the ViT.

It is a **hierarchical encoder inspired by the MovieGen Temporal Auto-Encoder (TAE)** ([github.com/MathieuTuli/MovieGen](https://github.com/MathieuTuli/MovieGen), `tae.py:740`), adapted to 4D fMRI.

**File**: `dinov2/layers/patch_embed_3d_plus_1d.py` — 4 classes :

| Line | Class                  | Role                                                                 |
|------|------------------------|----------------------------------------------------------------------|
| 29   | `Conv3Plus1d`          | Separable spatial-3D + temporal-1D conv (basic block of every level) |
| 68   | `_ResBlock3Plus1d`     | Residual block stacking 2× `Conv3Plus1d`                             |
| 85   | `PositionEmbedding3D`  | Factorised positional embedding (temporal + spatial + CLS)           |
| 117  | `PatchEmbed3DPlus1D`   | Top-level encoder stacking the 3 hierarchical levels                 |

\newpage

# 1. The basic building block: Conv3Plus1d

A **separable 4D convolution** = 3D spatial conv (per frame) + 1D temporal conv (per spatial position).

\begin{center}
\begin{tikzpicture}[
  node distance=0.4cm and 1.2cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=1.2cm, minimum width=2.6cm, align=center},
  arrow/.style={-{Stealth[length=3mm]}, thick},
  note/.style={text=cShape, font=\scriptsize\itshape},
]

\node[box, fill=cInput] (in) {Input\\(B, C, T, X, Y, Z)};

\node[box, fill=cConv, right=of in] (sp) {Spatial Conv3d\\$K_s{=}3$, $S_s{,}P_s$\\per-frame};

\node[box, fill=cConv, right=of sp] (tp) {Temporal Conv1d\\$K_t$, $S_t$, $P_t$\\per-voxel-pos};

\node[box, fill=cOut, right=of tp] (out) {Output\\(B, C', T', X', Y', Z')};

\draw[arrow] (in) -- (sp);
\draw[arrow] (sp) -- node[note, above]{rearrange} (tp);
\draw[arrow] (tp) -- (out);

\node[note, below=0.05cm of sp] {treats T as batch};
\node[note, below=0.05cm of tp] {treats X·Y·Z as batch};

\end{tikzpicture}
\end{center}

### Why "Plus" instead of full 4D conv

PyTorch has no native `Conv4d`. A full $(T, X, Y, Z)$ kernel of size $K^4$ would have $K^4 \cdot C^2$ parameters. The separable version $\text{Conv3d}(K^3) + \text{Conv1d}(K)$ has only $K^3 \cdot C^2 + K \cdot C^2 = (K^3 + K) \cdot C^2$ parameters — about **$K$ times fewer**.

This is the same trick as MovieGen's `Conv2Plus1d` (used in video) extended to 3D spatial.

### Equivalent PyTorch primitive

```python
class Conv3Plus1d(nn.Module):
    def __init__(self, in_c, out_c, K_s=3, S_s=1, P_s=1,
                 K_t=3, S_t=1, P_t=1):
        super().__init__()
        self.spatial  = nn.Conv3d(in_c, out_c, K_s, S_s, P_s)
        self.temporal = nn.Conv1d(out_c, out_c, K_t, S_t, P_t)

    def forward(self, x):  # x: (B, C, T, X, Y, Z)
        x = rearrange(x, 'b c t x y z -> (b t) c x y z')
        x = self.spatial(x)                    # 3D conv per frame
        x = rearrange(x, '(b t) c x y z -> (b x y z) c t', b=B, t=T)
        x = self.temporal(x)                   # 1D conv per voxel-pos
        x = rearrange(x, '(b x y z) c t -> b c t x y z', b=B, x=X, y=Y, z=Z)
        return x
```

\newpage

# 2. The residual block: _ResBlock3Plus1d

Two `Conv3Plus1d` stacked with GroupNorm + SiLU + residual connection.

\begin{center}
\begin{tikzpicture}[
  node distance=0.45cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=0.7cm, minimum width=3.4cm, align=center},
  arrow/.style={-{Stealth[length=3mm]}, thick},
]

\node[box, fill=cInput] (in) {Input \;\; (B, ch, T, X, Y, Z)};
\node[box, fill=cRes, below=of in] (n1) {GroupNorm + SiLU};
\node[box, fill=cConv, below=of n1] (c1) {Conv3Plus1d (ch $\to$ ch)};
\node[box, fill=cRes, below=of c1] (n2) {GroupNorm + SiLU};
\node[box, fill=cConv, below=of n2] (c2) {Conv3Plus1d (ch $\to$ ch)};
\node[circle, draw, fill=cOut, minimum size=0.7cm, below=of c2] (sum) {$+$};
\node[box, fill=cOut, below=of sum] (out) {Output \;\; (B, ch, T, X, Y, Z)};

\draw[arrow] (in) -- (n1);
\draw[arrow] (n1) -- (c1);
\draw[arrow] (c1) -- (n2);
\draw[arrow] (n2) -- (c2);
\draw[arrow] (c2) -- (sum);
\draw[arrow] (sum) -- (out);

\draw[arrow] (in.east) -- ++(1.4,0) |- (sum.east) node[midway, right, font=\scriptsize] {skip};

\end{tikzpicture}
\end{center}

Each ResBlock keeps the **same shape** (channels and spatial/temporal dimensions). Equivalent to MovieGen's `TemporalResnetBlock` (tae.py:643).

\newpage

# 3. Full encoder: PatchEmbed3DPlus1D

Three hierarchical levels chained with downsampling between them.

\begin{center}
\begin{tikzpicture}[
  node distance=0.42cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=0.9cm, minimum width=8.6cm, align=center},
  arrow/.style={-{Stealth[length=3mm]}, thick},
  note/.style={text=cShape, font=\scriptsize\itshape},
]

\node[box, fill=cInput] (in) {INPUT scan \quad (B, T, 1, X, Y, Z) \quad \textit{e.g.} (B, 1200, 1, 45, 54, 45)};

\node[box, fill=cConv, below=of in] (cin) {\textbf{conv\_in} \quad Conv3Plus1d \;\; 1 $\to$ 32 \\ spatial $K{=}3$, $S{=}3$ \;\;|\;\; temporal $K{=}3$, $S{=}1$};

\node[box, fill=cRes, below=of cin] (b0) {\textbf{block\_0} \quad ResBlock3Plus1d (32 $\to$ 32)};

\node[box, fill=cDown, below=of b0] (d0) {\textbf{down\_0} \quad Conv3Plus1d \;\; 32 $\to$ 64 \\ spatial $K{=}3$, $S{=}3$ \;\;|\;\; temporal $K{=}3$, $S{=}2$};

\node[box, fill=cRes, below=of d0] (b1) {\textbf{block\_1} \quad ResBlock3Plus1d (64 $\to$ 64)};

\node[box, fill=cDown, below=of b1] (d1) {\textbf{down\_1} \quad Conv3Plus1d \;\; 64 $\to$ 384 \\ spatial $K{=}3$, $S{=}1$ \;\;|\;\; temporal $K{=}K_t/2$, $S{=}K_t/2$};

\node[box, fill=cRes, below=of d1] (b2) {\textbf{block\_2} \quad ResBlock3Plus1d (384 $\to$ 384)};

\node[box, fill=cOut, below=of b2] (out) {OUTPUT token grid \quad (B, T\_eff × N\_spatial, 384) \\ rearrange \texttt{b c t x y z -> b (t x y z) c}};

\draw[arrow] (in) -- (cin);
\draw[arrow] (cin) -- (b0);
\draw[arrow] (b0) -- (d0);
\draw[arrow] (d0) -- (b1);
\draw[arrow] (b1) -- (d1);
\draw[arrow] (d1) -- (b2);
\draw[arrow] (b2) -- (out);

\end{tikzpicture}
\end{center}

\newpage

# 4. Shape evolution — concrete examples

### HCP-YA setup (T = 1200, temporal_kernel = 20)

| Stage      | Operation        | Output shape `(B, C, T, X, Y, Z)`   | Comment                              |
|------------|------------------|--------------------------------------|--------------------------------------|
| input      | —                | (B, 1, 1200, 45, 54, 45)             | raw 4D scan                          |
| conv_in    | spatial /3       | (B, 32, 1200, 15, 18, 15)            | first spatial downsample             |
| block_0    | identity         | (B, 32, 1200, 15, 18, 15)            | residual                             |
| down_0     | spatial /3, T /2 | (B, 64, 600, 5, 6, 5)                | total /9 spatial, /2 temporal        |
| block_1    | identity         | (B, 64, 600, 5, 6, 5)                | residual                             |
| down_1     | T /10            | (B, 384, 60, 5, 6, 5)                | total /9 spatial, /20 temporal       |
| block_2    | identity         | (B, 384, 60, 5, 6, 5)                | residual                             |
| **tokens** | rearrange        | **(B, 60 × 150, 384) = (B, 9000, 384)** | T_eff × N_spatial × embed_dim  |

### Mixed-T140 setup (T = 140, temporal_kernel = 14)

| Stage      | Operation        | Output shape                          | Comment                              |
|------------|------------------|---------------------------------------|--------------------------------------|
| input      | —                | (B, 1, 140, 45, 54, 45)               | raw 4D scan                          |
| conv_in    | spatial /3       | (B, 32, 140, 15, 18, 15)              |                                      |
| block_0    | identity         | (B, 32, 140, 15, 18, 15)              |                                      |
| down_0     | spatial /3, T /2 | (B, 64, 70, 5, 6, 5)                  |                                      |
| block_1    | identity         | (B, 64, 70, 5, 6, 5)                  |                                      |
| down_1     | T /7             | (B, 384, 10, 5, 6, 5)                 | $K_t = 14/2 = 7$                     |
| block_2    | identity         | (B, 384, 10, 5, 6, 5)                 |                                      |
| **tokens** | rearrange        | **(B, 10 × 150, 384) = (B, 1500, 384)** | 6× fewer tokens than HCP-YA   |

### Effective strides

- **Spatial total stride** = 3 × 3 × 1 = **9** → matches `patch_size = 9`
- **Temporal total stride** = 1 × 2 × $K_t$/2 = **$K_t$ = temporal_kernel**

\newpage

# 5. Factorised positional embedding (PositionEmbedding3D)

After the encoder, we add a **factorised** positional embedding before the ViT — three small tables broadcast and summed.

\begin{center}
\begin{tikzpicture}[
  node distance=0.5cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=0.9cm, minimum width=3.4cm, align=center},
  arrow/.style={-{Stealth[length=3mm]}, thick},
  note/.style={text=cShape, font=\scriptsize\itshape},
]

\node[box, fill=cInput] (toks) {Token grid \\ (B, T\_eff × N\_spatial, 384)};

\node[box, fill=cConv, right=2.6cm of toks, yshift=1.4cm] (pt) {pos\_temporal \\ (1, T\_eff, 384)};

\node[box, fill=cConv, right=2.6cm of toks] (ps) {pos\_spatial \\ (1, N\_spatial, 384)};

\node[box, fill=cConv, right=2.6cm of toks, yshift=-1.4cm] (pc) {pos\_cls \\ (1, 1, 384)};

\node[circle, draw, fill=cOut, minimum size=0.9cm, right=of ps] (plus) {$+$};

\node[box, fill=cOut, right=of plus] (out) {Tokens with pos \\ + CLS};

\draw[arrow] (toks) -- (plus.west) node[midway, above, note] {repeat \& sum};
\draw[arrow] (pt.east) -- (plus.north west);
\draw[arrow] (ps) -- (plus);
\draw[arrow] (pc.east) -- (plus.south west);
\draw[arrow] (plus) -- (out);

\end{tikzpicture}
\end{center}

### Parameter count comparison (HCP setup)

| Encoding | Shape                            | Params      |
|----------|----------------------------------|:-----------:|
| Flat 2D pos table (vanilla ViT) | (1, T_eff × N_spatial, 384) = (1, 9000, 384) | **3.46 M** |
| **Factorised (ours)** | pos_temporal (60×384) + pos_spatial (150×384) + pos_cls (1×384) | **80 K**   |

Factor of **~43× fewer parameters** for the positional embedding.

### Combine method (per scan)

```python
def combined_patch_pos(self):
    pos_t = repeat(self.pos_temporal, '1 t d -> 1 (t n) d', n=N_spatial)
    pos_s = repeat(self.pos_spatial,  '1 n d -> 1 (t n) d', t=T_eff)
    return pos_t + pos_s   # (1, T_eff*N_spatial, embed_dim)
```

The CLS token gets `pos_cls` added separately when prepended in `prepare_tokens_with_masks` of the ViT.

\newpage

# 6. Memory and compute notes

### Activation checkpointing

The full hierarchical encoder produces large intermediate activations (the level-0 tensor at full spatial resolution is `(B, 32, 1200, 15, 18, 15)` = ~16 GB for B=2 in fp32). We wrap the encoder forward in `torch.utils.checkpoint`:

```python
if self.training:
    x = torch.utils.checkpoint.checkpoint(
        self._encoder_forward, x, use_reentrant=False,
    )
else:
    x = self._encoder_forward(x)
```

Trade-off: **~30% extra forward compute** (re-computation during backward) for **~80 GB saved VRAM**.

### Parameter count

| Module               | Parameters    |
|----------------------|:-------------:|
| conv_in              | ~870          |
| block_0              | ~37 k         |
| down_0               | ~55 k         |
| block_1              | ~147 k        |
| down_1 (Kt=14)       | ~840 k        |
| block_2              | ~5.3 M        |
| PositionEmbedding3D  | ~80 k         |
| **Total**            | **~6.5 M**    |

(vs ~21 M for the 12 ViT-S transformer blocks downstream)

\newpage

# 7. Why this design

| Choice                                | Alternative                          | Why this one                                                                 |
|---------------------------------------|--------------------------------------|------------------------------------------------------------------------------|
| Separable Conv3+1D                    | Full 4D conv                         | PyTorch has no Conv4d; separable has $\sim K$ times fewer params              |
| **3 hierarchical levels** with downsample | Single big patchify conv         | Pyramidal feature hierarchy (à la TAE / Hiera) captures multi-scale structure |
| Conv3Plus1d in **every** level        | Spatial-only then temporal-only split| Avoids the "spatial collapse before temporal context" issue                  |
| GroupNorm + SiLU                      | BatchNorm + ReLU                     | BatchNorm needs batch stats, problematic with small batches / FSDP wrapping  |
| **Factorised** pos embed              | Flat 2D pos embed                    | $O(T_{\text{eff}} + N_{\text{spatial}})$ params instead of $O(T_{\text{eff}} \times N_{\text{spatial}})$  |
| temporal_kernel derived from config   | Hard-coded down_1 K_t                | Lets us train at $K_t=20$ (HCP) and $K_t=14$ (mixed) from same code          |
| Activation checkpointing on encoder   | No checkpointing                     | Encoder produces the largest intermediate tensors; ViT downstream is smaller |

# Reference

- **Original architecture inspiration**: MovieGen Temporal Auto-Encoder, `tae.py` lines 740 (`TemporalEncoder`), 573 (`Conv2Plus1d`), 643 (`TemporalResnetBlock`). See [github.com/MathieuTuli/MovieGen](https://github.com/MathieuTuli/MovieGen).
- **Code in this repo**: `dinov2/layers/patch_embed_3d_plus_1d.py`
- **Integration point**: `dinov2/models/__init__.py:build_model_from_cfg` injects `PatchEmbed3DPlus1D` as the `embed_layer` of `DinoVisionTransformer` when `cfg.student.fmri_mode = true`.
