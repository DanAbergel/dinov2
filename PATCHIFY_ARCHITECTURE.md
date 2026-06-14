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

**Hierarchical 3D+1D patchify encoder, inspired by the MovieGen TAE `TemporalEncoder`** ([tae.py:1052](https://github.com/MathieuTuli/MovieGen/blob/main/tae.py#L1052)).

\vspace{0.2cm}

## Building blocks

\begin{minipage}[t]{0.50\linewidth}
\textbf{Conv3Plus1d} — separable substitute for the missing PyTorch \texttt{Conv4d}. Splits a 4D conv into a 3D spatial conv (per frame) followed by a 1D temporal conv (per voxel-position).

\vspace{0.15cm}
\begin{center}
\begin{tikzpicture}[
  node distance=0.3cm,
  font=\scriptsize,
  box/.style={draw, rounded corners, minimum height=0.7cm, minimum width=3.2cm, align=center},
  arrow/.style={-{Stealth[length=2mm]}, thick},
]
\node[box, fill=cInput] (i) {Input \;\; (B, C, T, X, Y, Z)};
\node[box, fill=cConv, below=of i] (s) {Conv3d spatial $K{=}3$ \\ \scriptsize\itshape T merged into batch};
\node[box, fill=cConv, below=of s] (t) {Conv1d temporal $K{=}3$ \\ \scriptsize\itshape X·Y·Z merged into batch};
\node[box, fill=cOut, below=of t] (o) {Output \;\; (B, C', T', X', Y', Z')};
\draw[arrow] (i) -- (s); \draw[arrow] (s) -- (t); \draw[arrow] (t) -- (o);
\end{tikzpicture}
\end{center}
\end{minipage}\hfill
\begin{minipage}[t]{0.46\linewidth}
\textbf{ResBlock3Plus1d} — residual block of 2 \texttt{Conv3Plus1d} with GroupNorm + SiLU pre-activation, output added to a skip connection. Same shape in / out.

\vspace{0.15cm}
\begin{center}
\begin{tikzpicture}[
  node distance=0.22cm,
  font=\scriptsize,
  box/.style={draw, rounded corners, minimum height=0.55cm, minimum width=3cm, align=center},
  arrow/.style={-{Stealth[length=2mm]}, thick},
]
\node[box, fill=cInput] (i) {Input};
\node[box, fill=cRes, below=of i] (n1) {GroupNorm + SiLU};
\node[box, fill=cConv, below=of n1] (c1) {Conv3Plus1d};
\node[box, fill=cRes, below=of c1] (n2) {GroupNorm + SiLU};
\node[box, fill=cConv, below=of n2] (c2) {Conv3Plus1d};
\node[circle, draw, fill=cOut, minimum size=0.6cm, below=of c2] (sum) {$+$};
\node[box, fill=cOut, below=of sum] (o) {Output};
\draw[arrow] (i) -- (n1); \draw[arrow] (n1) -- (c1); \draw[arrow] (c1) -- (n2);
\draw[arrow] (n2) -- (c2); \draw[arrow] (c2) -- (sum); \draw[arrow] (sum) -- (o);
\draw[arrow] (i.east) -- ++(0.4,0) |- (sum.east) node[midway, right, font=\tiny] {skip};
\end{tikzpicture}
\end{center}
\end{minipage}

\vspace{0.1cm}

## Full encoder (HCP-YA setup, T = 1200, temporal_kernel = 20)

\begin{center}
\begin{tikzpicture}[
  node distance=0.3cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=0.7cm, minimum width=9.5cm, align=center},
  arrow/.style={-{Stealth[length=2.5mm]}, thick},
  shapelbl/.style={text=cNote, font=\scriptsize\itshape, right=0.2cm},
]
\node[box, fill=cInput] (in) {\textbf{INPUT scan}};
\node[shapelbl] at (in.east) {(B, 1200, 1, 45, 54, 45)};

\node[box, fill=cConv, below=of in] (cin) {\textbf{conv\_in} \quad Conv3Plus1d 1{$\to$}32, spatial $S{=}3$};
\node[shapelbl] at (cin.east) {(B, 32, 1200, 15, 18, 15)};

\node[box, fill=cRes, below=of cin] (b0) {\textbf{block\_0} \quad ResBlock3Plus1d (32{$\to$}32)};
\node[shapelbl] at (b0.east) {same shape};

\node[box, fill=cDown, below=of b0] (d0) {\textbf{down\_0} \quad Conv3Plus1d 32{$\to$}64, spatial $S{=}3$, temporal $S{=}2$};
\node[shapelbl] at (d0.east) {(B, 64, 600, 5, 6, 5)};

\node[box, fill=cRes, below=of d0] (b1) {\textbf{block\_1} \quad ResBlock3Plus1d (64{$\to$}64)};
\node[shapelbl] at (b1.east) {same shape};

\node[box, fill=cDown, below=of b1] (d1) {\textbf{down\_1} \quad Conv3Plus1d 64{$\to$}384, temporal $S{=}10$};
\node[shapelbl] at (d1.east) {(B, 384, 60, 5, 6, 5)};

\node[box, fill=cRes, below=of d1] (b2) {\textbf{block\_2} \quad ResBlock3Plus1d (384{$\to$}384)};
\node[shapelbl] at (b2.east) {same shape};

\node[box, fill=cOut, below=of b2] (out) {\textbf{OUTPUT tokens} \quad rearrange + factorised pos embed};
\node[shapelbl] at (out.east) {(B, 9000, 384)};

\draw[arrow] (in) -- (cin); \draw[arrow] (cin) -- (b0); \draw[arrow] (b0) -- (d0);
\draw[arrow] (d0) -- (b1); \draw[arrow] (b1) -- (d1); \draw[arrow] (d1) -- (b2);
\draw[arrow] (b2) -- (out);
\end{tikzpicture}
\end{center}

\vspace{0.1cm}

## Parameter breakdown

| Layer                     | What it contains                              | Params       |
|---------------------------|-----------------------------------------------|:------------:|
| `conv_in`                 | 1 Conv3Plus1d (1{$\to$}32)                    | 4 000        |
| `block_0`                 | 2 Conv3Plus1d (32{$\to$}32) + 2 GroupNorm     | 61 696       |
| `down_0`                  | 1 Conv3Plus1d (32{$\to$}64)                   | 67 712       |
| `block_1`                 | 2 Conv3Plus1d (64{$\to$}64) + 2 GroupNorm     | 246 272      |
| `down_1`                  | 1 Conv3Plus1d (64{$\to$}384)                  | 2 138 880    |
| `block_2`                 | 2 Conv3Plus1d (384{$\to$}384) + 2 GroupNorm   | 8 850 432    |
| `PositionEmbedding3D`     | `pos_temporal` + `pos_spatial` + `pos_cls`    | 81 024       |
| **TOTAL**                 |                                               | **11 450 016**  |

**Token grid output** : $T_\text{eff} = 60$, $N_\text{spatial} = 5 \cdot 6 \cdot 5 = 150$, total **9 000 tokens** of dim 384.

## Differences vs MovieGen TAE `TemporalEncoder`

| Aspect                   | MovieGen TAE                       | Ours (PatchEmbed3DPlus1D)            |
|--------------------------|------------------------------------|---------------------------------------|
| Domain                   | Video (T, H, W)                    | fMRI volumes (T, X, Y, Z)             |
| Building block           | `Conv2Plus1d` (2D spatial + 1D temporal) | `Conv3Plus1d` (3D spatial + 1D temporal) |
| Hierarchical levels      | 4 (default `ch_mult=(1,2,4,8)`)    | 3 (channels 32 / 64 / 384)            |
| ResBlocks per level      | 2                                  | 1                                     |
| Attention in encoder     | Yes (at specified resolutions)     | None                                  |
| Middle bottleneck section| 2 ResBlocks + 1 AttnBlock          | None                                  |
| Output                   | Latent tensor `z` (for decoder)    | Token sequence (B, $T_\text{eff} \cdot N_\text{spatial}$, 384) for ViT |
