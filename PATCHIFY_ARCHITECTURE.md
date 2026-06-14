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

**Hierarchical 3D+1D patchify encoder, inspired by MovieGen TAE** ([github.com/MathieuTuli/MovieGen](https://github.com/MathieuTuli/MovieGen), `tae.py:740`). Basic block = `Conv3Plus1d` = Conv3d spatial (per frame) + Conv1d temporal (per voxel-position) — separable substitute for the missing `Conv4d`.

\vspace{0.3cm}

\begin{center}
\begin{tikzpicture}[
  node distance=0.35cm,
  font=\footnotesize,
  box/.style={draw, rounded corners, minimum height=0.85cm, minimum width=9cm, align=center},
  arrow/.style={-{Stealth[length=2.5mm]}, thick},
  shapelbl/.style={text=cNote, font=\scriptsize\itshape, right=0.2cm},
]

\node[box, fill=cInput] (in) {\textbf{INPUT scan} \quad (B, T, 1, X, Y, Z)};
\node[shapelbl] at (in.east) {(B, 1200, 1, 45, 54, 45)};

\node[box, fill=cConv, below=of in] (cin) {\textbf{conv\_in} \quad Conv3Plus1d 1{$\to$}32 \;\; spatial $S{=}3$ \;\; \textbf{4\,k}};
\node[shapelbl] at (cin.east) {(B, 32, 1200, 15, 18, 15)};

\node[box, fill=cRes, below=of cin] (b0) {\textbf{block\_0} \quad ResBlock3Plus1d (32{$\to$}32) \;\; \textbf{62\,k}};
\node[shapelbl] at (b0.east) {same shape};

\node[box, fill=cDown, below=of b0] (d0) {\textbf{down\_0} \quad Conv3Plus1d 32{$\to$}64 \;\; spatial $S{=}3$, temp $S{=}2$ \;\; \textbf{68\,k}};
\node[shapelbl] at (d0.east) {(B, 64, 600, 5, 6, 5)};

\node[box, fill=cRes, below=of d0] (b1) {\textbf{block\_1} \quad ResBlock3Plus1d (64{$\to$}64) \;\; \textbf{246\,k}};
\node[shapelbl] at (b1.east) {same shape};

\node[box, fill=cDown, below=of b1] (d1) {\textbf{down\_1} \quad Conv3Plus1d 64{$\to$}384 \;\; temp $S{=}10$ \;\; \textbf{2.14\,M}};
\node[shapelbl] at (d1.east) {(B, 384, 60, 5, 6, 5)};

\node[box, fill=cRes, below=of d1] (b2) {\textbf{block\_2} \quad ResBlock3Plus1d (384{$\to$}384) \;\; \textbf{8.85\,M}};
\node[shapelbl] at (b2.east) {same shape};

\node[box, fill=cOut, below=of b2] (out) {\textbf{OUTPUT tokens} \quad rearrange + factorised pos embed \;\; \textbf{81\,k}};
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

\begin{center}
\textbf{Total patchify parameters : 11.45\,M}
\end{center}

\vspace{0.1cm}

**Setup HCP-YA** — $T{=}1200$, temporal_kernel ${=}20$, image $45 \times 54 \times 45$, patch_size ${=}9$.

**Strides totaux** : spatial $3 \cdot 3 \cdot 1 = 9$ (= patch_size), temporal $1 \cdot 2 \cdot 10 = 20$ (= temporal_kernel).

**Grille de tokens** : $T_\text{eff} = 60$, $N_\text{spatial} = 5 \cdot 6 \cdot 5 = 150$ → **9 000 tokens** dim 384.

**Pos embed factorisé** : `pos_temporal(60×384)` + `pos_spatial(150×384)` + `pos_cls(1×384)` = 81 k params, vs 3.46 M pour une table plate (43× moins).
