---
title: "Patchify architecture — PatchEmbed3DPlus1D"
date: "2026-06-14"
geometry: margin=1.5cm
fontsize: 9pt
header-includes:
  - \usepackage{tikz}
  - \usetikzlibrary{positioning, arrows.meta, fit, backgrounds, calc}
  - \definecolor{cInput}{RGB}{255,224,178}
  - \definecolor{cConv}{RGB}{179,205,255}
  - \definecolor{cRes}{RGB}{195,230,203}
  - \definecolor{cDown}{RGB}{255,179,179}
  - \definecolor{cOut}{RGB}{215,179,255}
  - \definecolor{cNote}{RGB}{100,100,100}
  - \pagestyle{empty}
---

**Hierarchical 3D+1D patchify encoder, inspired by MovieGen TAE** ([github.com/MathieuTuli/MovieGen](https://github.com/MathieuTuli/MovieGen), `tae.py:740`). Takes a 4D fMRI scan `(B, T, 1, X, Y, Z)` and outputs a flat sequence of `T_eff × N_spatial` tokens of dim 384, consumed by the ViT.

**Building block** = `Conv3Plus1d` (separable conv): Conv3d spatial (per frame) + Conv1d temporal (per voxel-position). PyTorch lacks a native Conv4d, so we replace it with this separable variant — same trick as MovieGen's `Conv2Plus1d`, ~$K$× fewer parameters than a full Conv4d.

\vspace{0.3cm}

\begin{center}
\begin{tikzpicture}[
  node distance=0.35cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=0.8cm, minimum width=9cm, align=center},
  arrow/.style={-{Stealth[length=2.5mm]}, thick},
  shapelbl/.style={text=cNote, font=\scriptsize\itshape, right=0.2cm},
]

\node[box, fill=cInput] (in) {\textbf{INPUT scan} \quad (B, T, 1, X, Y, Z)};
\node[shapelbl] at (in.east) {e.g.\ (B, 1200, 1, 45, 54, 45)};

\node[box, fill=cConv, below=of in] (cin) {\textbf{conv\_in} \quad Conv3Plus1d 1{$\to$}32 \;\; spatial $S{=}3$};
\node[shapelbl] at (cin.east) {(B, 32, 1200, 15, 18, 15)};

\node[box, fill=cRes, below=of cin] (b0) {\textbf{block\_0} \quad ResBlock3Plus1d (32{$\to$}32)};
\node[shapelbl] at (b0.east) {same shape};

\node[box, fill=cDown, below=of b0] (d0) {\textbf{down\_0} \quad Conv3Plus1d 32{$\to$}64 \;\; spatial $S{=}3$, temporal $S{=}2$};
\node[shapelbl] at (d0.east) {(B, 64, 600, 5, 6, 5)};

\node[box, fill=cRes, below=of d0] (b1) {\textbf{block\_1} \quad ResBlock3Plus1d (64{$\to$}64)};
\node[shapelbl] at (b1.east) {same shape};

\node[box, fill=cDown, below=of b1] (d1) {\textbf{down\_1} \quad Conv3Plus1d 64{$\to$}384 \;\; temporal $S{=}K_t/2$};
\node[shapelbl] at (d1.east) {(B, 384, 60, 5, 6, 5)};

\node[box, fill=cRes, below=of d1] (b2) {\textbf{block\_2} \quad ResBlock3Plus1d (384{$\to$}384)};
\node[shapelbl] at (b2.east) {same shape};

\node[box, fill=cOut, below=of b2] (out) {\textbf{OUTPUT tokens} \quad rearrange to (B, T\_eff $\cdot$ N\_spatial, 384) \;\;+ factorised pos embed};
\node[shapelbl] at (out.east) {(B, 9000, 384)};

\draw[arrow] (in) -- (cin);
\draw[arrow] (cin) -- (b0);
\draw[arrow] (b0) -- (d0);
\draw[arrow] (d0) -- (b1);
\draw[arrow] (b1) -- (d1);
\draw[arrow] (d1) -- (b2);
\draw[arrow] (b2) -- (out);

\end{tikzpicture}
\end{center}

\vspace{0.2cm}

**Total strides** — spatial $3 \cdot 3 \cdot 1 = 9$ (= patch_size), temporal $1 \cdot 2 \cdot K_t/2 = K_t$ (= temporal_kernel). Token grid is then $(T_\text{eff}, N_x, N_y, N_z)$ flattened.

**Factorised positional embedding** (before ViT) — three small tables broadcast and summed: `pos_temporal (T_eff)` + `pos_spatial (N_spatial)` + `pos_cls (1)`. For HCP setup: $O(60 + 150) \cdot 384 = 80$\,k params vs $O(60 \cdot 150) \cdot 384 = 3.46$\,M for a flat 2D pos table (\~43× fewer).

**Two configurations** —

| Setup        | T     | $K_t$ | $T_\text{eff}$ | $N_\text{spatial}$ | Total tokens |
|--------------|:-----:|:-----:|:--------------:|:------------------:|:------------:|
| HCP-YA       | 1200  | 20    | 60             | $5 \cdot 6 \cdot 5 = 150$ | **9000**     |
| Mixed-T140   | 140   | 14    | 10             | 150                | **1500**     |

**File**: `dinov2/layers/patch_embed_3d_plus_1d.py` (~230 lines, 4 classes: `Conv3Plus1d`, `_ResBlock3Plus1d`, `PositionEmbedding3D`, `PatchEmbed3DPlus1D`).
