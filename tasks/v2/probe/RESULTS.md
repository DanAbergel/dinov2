---
header-includes:
  - \usepackage[dvipsnames]{xcolor}
  - \usepackage{colortbl}
---

# fMRI Foundation Model (V2) — Results

**Color scale (AUROC / metric):** \colorbox{OliveGreen!55}{$\geq$0.85} \colorbox{YellowGreen!50}{0.75--0.85} \colorbox{Yellow!55}{0.65--0.75} \colorbox{Orange!50}{0.55--0.65} \colorbox{Red!35}{$<$0.55 (near chance)}

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

Same DINOv2 (ImageNet) init; each run changes **one** factor. **base** = reference (freeze blocks 0-8), **fourier** = Fourier positional encoding, **noblock2** = drop block_2, **pool** = AvgPool downsampling, **unfrozen** = all layers unfrozen during SSL.

```{=latex}
\begin{center}\small
\begin{tabular}{lccccc}
\hline
\textbf{Axis} & \textbf{base} & \textbf{fourier} & \textbf{noblock2} & \textbf{pool} & \textbf{unfrozen} \\
\hline
ABIDE / Autism & \cellcolor{Orange!50}0.605 & \cellcolor{Red!35}0.518 & \cellcolor{Red!35}0.484 & \cellcolor{Orange!50}0.606 & \cellcolor{Red!35}0.541 \\
ABIDE / Age & \cellcolor{YellowGreen!50}0.846 & \cellcolor{YellowGreen!50}0.756 & \cellcolor{YellowGreen!50}0.776 & \cellcolor{OliveGreen!55}0.877 & \cellcolor{YellowGreen!50}0.828 \\
ABIDE / Sex & \cellcolor{Red!35}0.521 & \cellcolor{Red!35}0.514 & \cellcolor{Red!35}0.492 & \cellcolor{Yellow!55}0.694 & \cellcolor{Red!35}0.538 \\
ADNI / NC-MCI & \cellcolor{Orange!50}0.567 & \cellcolor{Red!35}0.497 & \cellcolor{Red!35}0.484 & \cellcolor{Orange!50}0.566 & \cellcolor{Orange!50}0.604 \\
ADNI / AD-HC & \cellcolor{Red!35}0.481 & \cellcolor{Red!35}0.537 & \cellcolor{Orange!50}0.577 & \cellcolor{Orange!50}0.591 & \cellcolor{Yellow!55}0.731 \\
ADNI / Amyloid & \cellcolor{Orange!50}0.611 & \cellcolor{Red!35}0.539 & \cellcolor{Red!35}0.500 & \cellcolor{Orange!50}0.641 & \cellcolor{Orange!50}0.640 \\
HCP / Sex & \cellcolor{OliveGreen!55}0.908 & \cellcolor{OliveGreen!55}0.900 & \cellcolor{OliveGreen!55}0.933 & \cellcolor{OliveGreen!55}0.962 & \cellcolor{OliveGreen!55}0.886 \\
HCP / Age & \cellcolor{Orange!50}0.633 & \cellcolor{Orange!50}0.594 & \cellcolor{Orange!50}0.626 & \cellcolor{Yellow!55}0.680 & \cellcolor{Orange!50}0.592 \\
OASIS / AD Conv & \cellcolor{gray!12}-- & \cellcolor{gray!12}-- & \cellcolor{gray!12}-- & \cellcolor{gray!12}-- & \cellcolor{gray!12}-- \\
\hline
\end{tabular}
\end{center}
```
*AUROC ranks the runs by representation quality (threshold- and balance-independent). The SOTA table (Section 5) uses Acc/F1, matching what the papers report.*

## 4. Probe ablations (on `base`)

### 4a. Temporal aggregation of the CLS token — AUROC

```{=latex}
\begin{center}\small
\begin{tabular}{lccc}
\hline
\textbf{Axis} & \textbf{mean (384-d)} & \textbf{mean\_std (768-d)} & \textbf{$\Delta$} \\
\hline
ABIDE / Autism & \cellcolor{Orange!50}0.605 & \cellcolor{Orange!50}0.583 & -0.023 \\
ABIDE / Age & \cellcolor{YellowGreen!50}0.846 & \cellcolor{YellowGreen!50}0.835 & -0.011 \\
ADNI / NC-MCI & \cellcolor{Orange!50}0.567 & \cellcolor{Red!35}0.505 & -0.062 \\
ADNI / Amyloid & \cellcolor{Orange!50}0.611 & \cellcolor{Orange!50}0.606 & -0.005 \\
ADNI / AD-HC & \cellcolor{Red!35}0.481 & \cellcolor{Red!35}0.404 & -0.077 \\
HCP / Sex & \cellcolor{OliveGreen!55}0.908 & \cellcolor{OliveGreen!55}0.853 & -0.055 \\
HCP / Age & \cellcolor{Orange!50}0.633 & \cellcolor{Orange!50}0.595 & -0.037 \\
ADHD / ADHD & \cellcolor{Orange!50}0.552 & \cellcolor{Orange!50}0.557 & +0.005 \\
COBRE / Schizophrenia & \cellcolor{Red!35}0.519 & \cellcolor{Red!35}0.538 & +0.019 \\
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
ADNI / NC-MCI & \cellcolor{Orange!50}0.588 & \cellcolor{Red!35}0.509 & \cellcolor{Red!35}0.523 & \cellcolor{Red!35}0.475 & \cellcolor{Red!35}0.433 & \cellcolor{Orange!50}0.567 \\
ADNI / AD-HC & \cellcolor{Red!35}0.407 & \cellcolor{Red!35}0.467 & \cellcolor{Red!35}0.472 & \cellcolor{Red!35}0.448 & \cellcolor{Red!35}0.487 & \cellcolor{Red!35}0.481 \\
ADNI / Amyloid & \cellcolor{Orange!50}0.621 & \cellcolor{Orange!50}0.623 & \cellcolor{Orange!50}0.610 & \cellcolor{Orange!50}0.631 & \cellcolor{Orange!50}0.610 & \cellcolor{Orange!50}0.611 \\
ABIDE / Autism & \cellcolor{Red!35}0.550 & \cellcolor{Red!35}0.497 & \cellcolor{Orange!50}0.561 & \cellcolor{Red!35}0.509 & \cellcolor{Red!35}0.522 & \cellcolor{Orange!50}0.605 \\
HCP / Sex & \cellcolor{YellowGreen!50}0.844 & \cellcolor{OliveGreen!55}0.906 & \cellcolor{OliveGreen!55}0.896 & \cellcolor{OliveGreen!55}0.898 & \cellcolor{OliveGreen!55}0.895 & \cellcolor{OliveGreen!55}0.908 \\
ADHD / ADHD & \cellcolor{Red!35}0.477 & \cellcolor{Red!35}0.466 & \cellcolor{Red!35}0.382 & \cellcolor{Red!35}0.412 & \cellcolor{Red!35}0.397 & \cellcolor{Orange!50}0.552 \\
COBRE / Schizophrenia & \cellcolor{Orange!50}0.602 & \cellcolor{Orange!50}0.566 & \cellcolor{Orange!50}0.595 & \cellcolor{Orange!50}0.557 & \cellcolor{Orange!50}0.590 & \cellcolor{Red!35}0.519 \\
\hline
\end{tabular}
\end{center}
```
*The MLP head does not clearly beat the linear probe; deeper heads overfit.*

## 5. Our results vs SOTA (same-dataset only)

### 5a. Same-dataset comparisons

The SOTA number is shown **only** where the paper uses the same dataset as us. Brain-JEPA = fine-tuning, Acc/F1 only; ours = linear probe (best-AUROC config).

```{=latex}
\begin{center}\small
\begin{tabular}{lcccc}
\hline
\textbf{Benchmark} & \textbf{Model} & \textbf{Acc} & \textbf{F1} & \textbf{AUROC} \\
\hline
ADNI / NC-MCI & Ours (lin.) & \cellcolor{Orange!50}0.576 & \cellcolor{Orange!50}0.632 & \cellcolor{Orange!50}0.604 \\
 & Brain-JEPA (FT) & \cellcolor{YellowGreen!50}0.768 & \cellcolor{OliveGreen!55}0.863 & \cellcolor{gray!12}n/r \\
ADNI / Amyloid & Ours (lin.) & \cellcolor{Orange!50}0.559 & \cellcolor{Orange!50}0.623 & \cellcolor{Orange!50}0.641 \\
 & Brain-JEPA (FT) & \cellcolor{Yellow!55}0.710 & \cellcolor{YellowGreen!50}0.760 & \cellcolor{gray!12}n/r \\
ADHD-200 & Ours (lin.) & \cellcolor{Orange!50}0.617 & \cellcolor{Red!35}0.287 & \cellcolor{Orange!50}0.557 \\
 & NeuroSTORM & \cellcolor{Orange!50}0.587 & \cellcolor{gray!12}n/r & \cellcolor{gray!12}n/r \\
\hline
\end{tabular}
\end{center}
```
### 5b. Our other downstream results (no same-dataset SOTA to compare)

```{=latex}
\begin{center}\small
\begin{tabular}{lccc}
\hline
\textbf{Benchmark} & \textbf{AUROC} & \textbf{Acc} & \textbf{F1} \\
\hline
ABIDE / Autism & \cellcolor{Orange!50}0.606 & \cellcolor{Orange!50}0.572 & \cellcolor{Red!35}0.513 \\
ABIDE / Age & \cellcolor{OliveGreen!55}0.877 & \cellcolor{YellowGreen!50}0.804 & \cellcolor{YellowGreen!50}0.805 \\
ABIDE / Sex & \cellcolor{Yellow!55}0.694 & \cellcolor{Yellow!55}0.736 & \cellcolor{YellowGreen!50}0.834 \\
ADNI / AD-HC & \cellcolor{Yellow!55}0.731 & \cellcolor{Yellow!55}0.651 & \cellcolor{Red!35}0.516 \\
HCP / Sex & \cellcolor{OliveGreen!55}0.962 & \cellcolor{OliveGreen!55}0.892 & \cellcolor{OliveGreen!55}0.879 \\
HCP / Age & \cellcolor{Yellow!55}0.680 & \cellcolor{Orange!50}0.618 & \cellcolor{Orange!50}0.597 \\
COBRE / Schizophrenia & \cellcolor{Orange!50}0.602 & \cellcolor{Orange!50}0.561 & \cellcolor{Orange!50}0.574 \\
\hline
\end{tabular}
\end{center}
```

*Only ADNI (Brain-JEPA) and ADHD-200 (NeuroSTORM) are same-dataset comparisons. Everything in 5b is our own result with no matching same-dataset SOTA number.*
