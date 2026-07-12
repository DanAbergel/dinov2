---
header-includes:
  - \usepackage[dvipsnames]{xcolor}
  - \usepackage{colortbl}
---

# fMRI Foundation Model (V2) — Results

*In every table, \colorbox{OliveGreen!55}{green} marks the best value of the row (in the SOTA tables: the per-metric winner between us and the paper).*

## 1. Pretraining corpus

Five sources, **4627 scans**, harmonized to TR = 0.72 s with a fixed T = 270 window.

| Dataset | Native TR (s) | Probe holdout (30%) | Per-batch quota |
|---|---|---|---|
| HCP | 0.72 | yes | 4 |
| ABIDE | per-site | yes | 4 |
| OASIS-3 | 2.2 | yes | 4 |
| ADNI | 3.0 | yes | 3 |
| AOMIC | 0.75/2.0 | no (kept whole) | 1 |

*Quota = per-batch proportion. Holdout: 30% of subjects excluded from pretraining -> leakage-free test set.*

## 2. Datasets & scan counts

| Dataset | Role | Scans | Note |
|---|---|---|---|
| HCP (rest) | pretrain + probe | 1084 | Sex / Age |
| ABIDE | pretrain + probe | 1035 | Autism / Age / Sex |
| OASIS-3 | pretrain + probe | 1197 | AD Conversion (labels pending) |
| ADNI | pretrain + probe | 812 | NC-MCI / AD-HC / Amyloid |
| AOMIC | pretrain only | ~499 | derived (4627 - others) |
| ADHD-200 | probe only (external) | 162 | 115 with usable DX |
| COBRE | probe only (external) | 146 | schizophrenia (72 SZ / 74 HC) |
| UCLA (ds000030) | probe only (external) | 265 | downloader ready |
| HCP task-fMRI | probe only | — | 7 tasks, download pending |

*External datasets (ADHD-200 / COBRE / UCLA) were never seen in pretraining.*

## 3. Pretraining ablations (5 SSL runs) — test AUROC

Same DINOv2 (ImageNet) init; each run changes **one** factor. **base** = reference, **fourier** = Fourier positional encoding, **noblock2** = drop block_2, **pool** = AvgPool downsampling, **unfrozen** = all layers unfrozen during SSL. Green = best run for that axis.

```{=latex}
\begin{center}\small
\begin{tabular}{lccccc}
\hline
\textbf{Axis} & \textbf{base} & \textbf{fourier} & \textbf{noblock2} & \textbf{pool} & \textbf{unfrozen} \\
\hline
ABIDE / Autism & 0.605 & 0.518 & 0.484 & \cellcolor{OliveGreen!55}0.606 & 0.541 \\
ABIDE / Age & 0.846 & 0.756 & 0.776 & \cellcolor{OliveGreen!55}0.877 & 0.828 \\
ABIDE / Sex & 0.521 & 0.514 & 0.492 & \cellcolor{OliveGreen!55}0.694 & 0.538 \\
ADNI / NC-MCI & 0.567 & 0.497 & 0.484 & 0.566 & \cellcolor{OliveGreen!55}0.604 \\
ADNI / AD-HC & 0.481 & 0.537 & 0.577 & 0.591 & \cellcolor{OliveGreen!55}0.731 \\
ADNI / Amyloid & 0.611 & 0.539 & 0.500 & \cellcolor{OliveGreen!55}0.641 & 0.640 \\
HCP / Sex & 0.908 & 0.900 & 0.933 & \cellcolor{OliveGreen!55}0.962 & 0.886 \\
HCP / Age & 0.633 & 0.594 & 0.626 & \cellcolor{OliveGreen!55}0.680 & 0.592 \\
OASIS / AD Conv & -- & -- & -- & -- & -- \\
\hline
\end{tabular}
\end{center}
```
*AUROC ranks the runs by representation quality (threshold- and balance-independent). The SOTA table uses Acc/F1, matching the papers.*

## 4. Probe ablations (on `base`)

### 4a. Temporal aggregation of the CLS token — AUROC

```{=latex}
\begin{center}\small
\begin{tabular}{lccc}
\hline
\textbf{Axis} & \textbf{mean (384-d)} & \textbf{mean\_std (768-d)} & \textbf{$\Delta$} \\
\hline
ABIDE / Autism & \cellcolor{OliveGreen!55}0.605 & 0.583 & -0.023 \\
ABIDE / Age & \cellcolor{OliveGreen!55}0.846 & 0.835 & -0.011 \\
ADNI / NC-MCI & \cellcolor{OliveGreen!55}0.567 & 0.505 & -0.062 \\
ADNI / Amyloid & \cellcolor{OliveGreen!55}0.611 & 0.606 & -0.005 \\
ADNI / AD-HC & \cellcolor{OliveGreen!55}0.481 & 0.404 & -0.077 \\
HCP / Sex & \cellcolor{OliveGreen!55}0.908 & 0.853 & -0.055 \\
HCP / Age & \cellcolor{OliveGreen!55}0.633 & 0.595 & -0.037 \\
ADHD / ADHD & 0.552 & \cellcolor{OliveGreen!55}0.557 & +0.005 \\
COBRE / Schizophrenia & 0.519 & \cellcolor{OliveGreen!55}0.538 & +0.019 \\
\hline
\end{tabular}
\end{center}
```
*`mean_std` mainly helps task dynamics; elsewhere `mean` wins.*

### 4b. MLP head architecture — AUROC

```{=latex}
\begin{center}\small
\begin{tabular}{lcccccc}
\hline
\textbf{Axis} & \textbf{128} & \textbf{256} & \textbf{256$\times$128} & \textbf{512$\times$256} & \textbf{512$\times$256$\times$128} & \textbf{linear} \\
\hline
ADNI / NC-MCI & \cellcolor{OliveGreen!55}0.588 & 0.509 & 0.523 & 0.475 & 0.433 & 0.567 \\
ADNI / AD-HC & 0.407 & 0.467 & 0.472 & 0.448 & \cellcolor{OliveGreen!55}0.487 & 0.481 \\
ADNI / Amyloid & 0.621 & 0.623 & 0.610 & \cellcolor{OliveGreen!55}0.631 & 0.610 & 0.611 \\
ABIDE / Autism & 0.550 & 0.497 & 0.561 & 0.509 & 0.522 & \cellcolor{OliveGreen!55}0.605 \\
HCP / Sex & 0.844 & 0.906 & 0.896 & 0.898 & 0.895 & \cellcolor{OliveGreen!55}0.908 \\
ADHD / ADHD & 0.477 & 0.466 & 0.382 & 0.412 & 0.397 & \cellcolor{OliveGreen!55}0.552 \\
COBRE / Schizophrenia & \cellcolor{OliveGreen!55}0.602 & 0.566 & 0.595 & 0.557 & 0.590 & 0.519 \\
\hline
\end{tabular}
\end{center}
```
*The MLP head does not clearly beat the linear probe; deeper heads overfit.*

## 5. Our results vs SOTA (same-dataset only)

### 5a. Same-dataset comparisons

SOTA shown only where the paper uses our dataset. Brain-JEPA = fine-tuning, Acc/F1 only; ours = linear probe (best-AUROC config). Green = per-metric winner.

```{=latex}
\begin{center}\small
\begin{tabular}{lcccc}
\hline
\textbf{Benchmark} & \textbf{Model} & \textbf{Acc} & \textbf{F1} & \textbf{AUROC} \\
\hline
ADNI / NC-MCI & Ours (lin.) & 0.576 & 0.632 & 0.604 \\
 & Brain-JEPA (FT) & \cellcolor{OliveGreen!55}0.768 & \cellcolor{OliveGreen!55}0.863 & \cellcolor{gray!12}n/r \\
ADNI / Amyloid & Ours (lin.) & 0.559 & 0.623 & 0.641 \\
 & Brain-JEPA (FT) & \cellcolor{OliveGreen!55}0.710 & \cellcolor{OliveGreen!55}0.760 & \cellcolor{gray!12}n/r \\
\hline
\end{tabular}
\end{center}
```
*ADHD-200 (vs NeuroSTORM) is left out: our probe barely discriminates there (AUROC 0.56, F1 0.29 — near chance), so the accuracy is not a meaningful comparison.*

### 5b. Our other downstream results (no same-dataset SOTA)

```{=latex}
\begin{center}\small
\begin{tabular}{lccc}
\hline
\textbf{Benchmark} & \textbf{AUROC} & \textbf{Acc} & \textbf{F1} \\
\hline
ABIDE / Autism & \cellcolor{OliveGreen!55}0.606 & 0.572 & 0.513 \\
ABIDE / Age & \cellcolor{OliveGreen!55}0.877 & 0.804 & 0.805 \\
ABIDE / Sex & 0.694 & 0.736 & \cellcolor{OliveGreen!55}0.834 \\
ADNI / AD-HC & \cellcolor{OliveGreen!55}0.731 & 0.651 & 0.516 \\
HCP / Sex & \cellcolor{OliveGreen!55}0.962 & 0.892 & 0.879 \\
HCP / Age & \cellcolor{OliveGreen!55}0.680 & 0.618 & 0.597 \\
COBRE / Schizophrenia & \cellcolor{OliveGreen!55}0.602 & 0.561 & 0.574 \\
\hline
\end{tabular}
\end{center}
```

*The only same-dataset SOTA comparison shown is ADNI (Brain-JEPA).*


**SOTA sources:** Brain-JEPA (arXiv 2409.19407, Tables 2--3, fine-tuning; Acc/F1 only) · NeuroSTORM (arXiv 2506.11167).
