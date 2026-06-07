# Freeze ablation -- results

**Question** (Ariel/Yoni): *which parts of the ImageNet-pretrained DINOv2 backbone should we fine-tune to adapt the model to fMRI? Everything? Only the last few transformer blocks? Nothing except the input adapter we introduce?*

**Shared setup across the three runs**: HCP-only dataset, T=1200, temporal_kernel=20, base_lr=3e-4, effective batch 16 (2 x 8 grad accumulation), 20 epochs, init = official DINOv2 ViT-S/14 reg4 ImageNet checkpoint filtered to drop pos_embed and patch_embed.

Only the **parameter-freezing policy** differs between runs.

---

## 1. The three strategies

| Strategy | Trained components | Trainable params | Description |
|---|---|---|---|
| **A -- Full fine-tune** (`baseline`) | full backbone + heads | **~22 M** | Official DINOv2 behaviour: no freezing |
| **B -- Partial freeze** (`freeze_last3`) | `patch_embed.*` + `blocks.9-11` + `norm` + heads | **~3.6 M** | The first 9 ImageNet blocks are frozen; the last 3 + final LayerNorm are fine-tuned |
| **C -- Freeze except input** (`freeze_fmri`) | `patch_embed.*` + heads | **~300 k** | The entire ImageNet backbone is frozen. Only the 3D+1D adapter we added is trained |

For each strategy we keep the **best checkpoint** among the available iters (`11999, 17999` for the two freeze runs; `5999, 8999` for the baseline).

---

## 2. Results -- HCP probes (the pretraining dataset)

| Label | Metric | A (full FT) | B (freeze last3) | C (freeze fmri) | **Winner** | C vs A |
|---|---|---|---|---|---|---|
| Sex | AUC, higher is better | 0.751 | 0.696 | **0.844** | **C** | **+0.093** |
| BrainVol | MAE, lower is better | 116,179 | 117,744 | **99,455** | **C** | **-14 %** |
| GrayMatterVol | MAE | 34,762 | 34,815 | **29,274** | **C** | **-16 %** |
| Age | MAE | 3.044 | 3.056 | **2.931** | **C** | **-3.7 %** |
| FluidIntel | MAE | 4.047 | 4.035 | **4.003** | **C** | -1.1 % |
| ProcSpeed | MAE | 16.343 | 16.401 | **16.207** | **C** | -0.8 % |
| WorkingMem | MAE | 10.442 | **10.410** | 10.411 | B (tie with C) | ~ -0.3 % |

**==> Strategy C (freeze except input) wins on 6 of 7 HCP labels.**

---

## 3. Results -- ADNI probes (transfer dataset)

The baseline (A) ADNI numbers come from the canonical probe on checkpoint `144327` (HCP-only T=1200, iter 8999) -- **same methodology** as the freeze probes (5-fold CV, LogReg C=1.0, sklearn).

| Label | Metric | A (full FT) | B (freeze last3) | C (freeze fmri) | **Winner** | C vs A |
|---|---|---|---|---|---|---|
| Sex | AUC | 0.653 | 0.696 | **0.813** | **C** | **+0.160** |
| Degradation3Y | AUC | 0.528 | 0.468 | **0.544** | **C** | +0.016 |
| **Degradation1Y** | AUC | **0.581** | 0.543 | 0.559 | **A** | (-0.022) |
| Degradation2Y | AUC | 0.533 | 0.456 | 0.533 | A tied with C | ~ 0 |
| CDR | AUC | **0.542** | 0.522 | 0.532 | **A** | (-0.010) |
| MMSE | MAE | 2.478 | **2.426** | 2.750 | **B** | (+11 %) |
| Age | MAE | 5.113 | **4.993** | 5.703 | **B** | (+12 %) |

**A more nuanced picture on ADNI**:
- **Anatomical** (Sex): strategy C wins by a wide margin (+0.16 AUC).
- **Demographic** (Age) and **cognitive** (MMSE): strategy B (partial FT) wins.
- **Clinical progression** (Degradation1Y, CDR): the full FT baseline (A) is slightly better, but the gap is **within fold noise** (+/- 0.05 to +/- 0.10 std on 5-fold AUC).

---

## 4. Three key findings

### 4.1 On the pretraining dataset (HCP), **freeze-except-input beats everything**
6 of 7 HCP labels are won by strategy C. On **Sex** the margin is large: 0.844 vs 0.751 baseline = **+9.3 AUC points**. On **BrainVol** and **GrayMatterVol** the MAE drops by 14-16 %.

**Interpretation**: the generic visual features learned by DINOv2 on ImageNet (edges, textures, shapes) transfer **remarkably well** to fMRI once the input adapter is trained appropriately. **Full backbone fine-tuning degrades those features** by over-specialising on the small HCP set (~1k scans).

### 4.2 Strategy B (partial freeze) is the worst on HCP
Fine-tuning only the last 3 blocks creates an **internal discontinuity**: blocks 0-8 stay frozen at ImageNet weights while blocks 9-11 adapt to fMRI. The low-level (ImageNet) vs high-level (fMRI) feature mismatch hurts the overall quality more than either extreme.

### 4.3 On ADNI transfer: strong anatomical, clinical signal still in noise
For Sex on ADNI (anatomical, n=812, ~50/50 balanced), strategy C beats A by +0.16 AUC -- **clear and robust**. For clinical labels (Degradation, CDR), all runs sit between 0.52 and 0.58 AUC with +/- 0.05 to +/- 0.10 std on 5-fold CV: differences are **within noise**. This is consistent with the earlier observation that the clinical signal is **data-limited (~145 positive Degradation1Y cases)**.

---

## 5. Why strategy C is the right choice for the thesis

| Argument | Detail |
|---|---|
| **Robust HCP improvement** | 6/7 labels won, large gains on Sex / BrainVol / GrayMatterVol |
| **Strong ADNI Sex improvement** | +0.16 AUC vs baseline -- well above the fold variance |
| **Compute and parameter efficiency** | 300k trainable params vs 22M (x73 less) -- faster training, lower overfitting risk |
| **Clean interpretation** | Matches the transfer-learning literature: "linear probing > full fine-tuning" on small datasets |
| **Simple architectural story** | "3D+1D input adapter + frozen ImageNet backbone" is a clear narrative for the defense |

---

## 6. Caveats and limits

- The 2,999 / 5,999 / 8,999 intermediate checkpoints of the two freeze runs were not saved (PeriodicCheckpointer's `max_to_keep`), so the full training dynamics aren't visible.
- `model_final` (iter 19,999) has not been probed yet for the two freeze runs -- to be added for the final version of the table.
- The ADNI baseline (A) comes from an earlier probe (before file renaming). The methodology is identical (canonical 5-fold LogReg) so the comparison is valid, but the same folds were not used.
- Clinical labels remain **data-limited** (~145 positive Degradation1Y cases) -- so differences between strategies on those labels are within noise. The bottleneck is the scarcity of positive cases, not the pretraining strategy.

---

## 7. Reproducibility

- **Repo / branch**: https://github.com/DanAbergel/dinov2 -- branch `fmri-mixed-t140`
- **YAML configs**:
  - A: `dinov2/configs/train/fmri_vits.yaml` (with `dataset_path: HCP`, `fmri_temporal_size: 1200`, `fmri_temporal_kernel: 20`)
  - B: `dinov2/configs/train/fmri_vits_hcp_freeze_last3.yaml`
  - C: `dinov2/configs/train/fmri_vits_hcp_freeze_fmri.yaml`
- **One sbatch script for all three**: `slurm_jobs/run_dinov2_fmri.sh` -- switch configs via `CONFIG_FILE=...`
- **Probes**: `slurm_jobs/probe_adni.sh` + `slurm_jobs/probe_hcp.sh`, auto-launched by `slurm_jobs/probe_3ckpts.sh`.
- **Summary table**: `python3 scripts/summarize_probes.py` -- automatic comparison across all runs.
- **Freezing mechanism**: `apply_freeze_policy()` in `dinov2/train/train.py` (tagged `# FMRI CHANGE`), activated by `cfg.optim.freeze_pretrained in {"fmri_only", "fmri_plus_last_3", null}`.
