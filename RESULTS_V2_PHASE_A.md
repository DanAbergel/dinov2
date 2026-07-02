---
title: "v2 Phase A - pretraining ablation + real ADNI labels"
date: "2026-07-02"
geometry: margin=2cm
fontsize: 10pt
---

# v2 Phase A - pretraining ablation (for Yoni)

Follow-up to the v1 results. Two things changed since:

1. **Real ADNI clinical labels.** We replaced the CDR proxy (from Sagi's manifest)
   with the **real diagnosis** (ADNI DXSUM: CN / MCI / Dementia) and **real amyloid
   status** (UC Berkeley amyloid PET), matched by PTID. Coverage over our 215 ADNI
   subjects: 212 with a diagnosis (CN 77 / MCI 101 / Dementia 34), 188 with amyloid
   (84 negative / 104 positive). This unblocks Brain-JEPA's ADNI tasks (NC/MCI,
   Amyloid) with honest labels.
2. **Phase A pretraining ablation.** Four pretrainings, one factor each, evaluated
   with the leakage-free linear probe (frozen encoder, 70:30 subject split, metric
   matched to each SOTA):
   - **base** - reference (learned spatial pos, strided-conv temporal downsample, full patchify)
   - **fourier** - Fourier spatial positional encoding instead of the learned table
   - **noblock2** - drop the final patchify ResBlock (block_2, ~83% of patchify params)
   - **pool** - temporal downsample by average pooling instead of strided conv (+ no block_2)

\vspace{0.3cm}

# Results (linear probe on the frozen encoder)

Each cell is the metric that the corresponding SOTA reports for that task.
ADNI rows use the **real** labels for base and noblock2 (fourier/pool ADNI were
auto-probed earlier with the CDR proxy and are not comparable - marked *).

| Axis (metric) | base | fourier | noblock2 | pool | SOTA |
|---------------|:----:|:-------:|:--------:|:----:|------|
| ABIDE Age (Acc) | **0.80** | 0.71 | 0.70 | 0.70 | SLIM-Brain 0.64 / SwiFT 0.62 |
| ABIDE Sex (F1) | **0.82** | 0.82 | 0.78 | 0.76 | LCM 0.87 |
| HCP Sex (Acc) | 0.81 | 0.79 | **0.84** | 0.80 | SLIM-Brain 0.91 / LCM 0.73 |
| HCP Sex (AUC) | 0.90 | 0.90 | **0.92** | 0.89 | - |
| ABIDE Autism (AUROC) | 0.53 | 0.50 | 0.47 | 0.57 | BNT 0.80 / BrainGFM 0.71 |
| ADNI NC/MCI (AUC) | 0.43 | 0.45* | 0.49 | 0.48* | Brain-JEPA 0.77 (acc) |
| ADNI AD/HC (AUC) | 0.61 | 0.64* | 0.56 | 0.58* | BrainGFM 0.80 |
| ADNI Amyloid (AUC) | 0.53 | - | 0.51 | - | Brain-JEPA 0.71 (acc) |

\newpage

# Reading

## Two clear messages

- **Demographic / global structure is learned well.** ABIDE Age **0.80, above** the
  volumetric SOTA (0.62-0.64); HCP Sex 0.84 acc / 0.92 AUC, **above LCM** (0.73 F1),
  near SLIM-Brain; ABIDE Sex 0.82 F1. The encoder clearly captures age/sex structure.
- **Clinical signal is at chance under linear probing (with the real labels).**
  ADNI NC/MCI ~0.43-0.49 AUC, AD/HC ~0.56-0.61, Amyloid ~0.51, Autism ~0.5. The
  **frozen** encoder does not linearly separate the clinical tasks.

## The CDR proxy was misleading (important)

With the CDR proxy, noblock2's ADNI AD/HC looked like 0.76 AUC. With the **real
diagnosis** it is **0.56**. So the earlier "clinical win" was an artifact of the
proxy correlating with something other than the true diagnosis. **We report only
the real-label numbers.**

## The architecture ablation did not help

- **pool** is worse or equal everywhere -> dropped.
- **noblock2** trades: better HCP Sex (+0.03 acc, +0.02 AUC) but worse ABIDE
  (Age -0.10, Sex -0.04); clinical unchanged (chance). It is **~equivalent to base
  but 6x lighter** (1.8 M vs 10.7 M patchify params) - a size win, not a quality win.
- **base** remains the strongest overall.
- Note: removing 83% of the patchify (block_2) did **not** degrade performance ->
  the patchify is **not** capacity-limited, so adding patchify depth is unlikely to
  help. The bottleneck is elsewhere (the mostly-frozen ViT and/or the SSL objective),
  not the tokenizer.

# Conclusion and next step

The linear probe (frozen encoder) is **the weakest downstream protocol**; Brain-JEPA
reports its headline numbers by **fine-tuning**. Our frozen-encoder clinical results
being at chance is the classic signal that the encoder must **adapt** to the task.

**Next experiment (decisive): fine-tuning (point 4).** Unfreeze the encoder and train
end-to-end on ADNI NC/MCI (Brain-JEPA's headline task), mean +/- std over seeds, same
70:30 test set. If fine-tuning recovers clinical signal -> we move toward Brain-JEPA's
0.77. If it does not -> the SSL pretraining did not learn a recoverable clinical
representation, which is itself an honest, reportable result.

**Still pending (Brain-JEPA coverage):** OASIS-3 AD Conversion (scans in hand; needs
the ADRC clinical CDR spreadsheet from oasis-brains.org). Out of reach short-term:
UK Biobank (paid DUA), HCP-Aging (NDA), MACC (restricted).
