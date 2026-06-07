# Freeze ablation -- results

Three runs sharing the same setup (HCP-only, T=1200, temporal_kernel=20, base_lr=3e-4, effective batch 16, 20 epochs, DINOv2 ViT-S/14 reg4 ImageNet init). Only the parameter-freezing policy differs.

## Strategies

| Strategy | Trained components | Trainable params |
|---|---|---|
| **A -- Full fine-tune** | full backbone + heads | ~22 M |
| **B -- Freeze last3** | `patch_embed.*` + `blocks.9-11` + `norm` + heads | ~3.6 M |
| **C -- Freeze except input** | `patch_embed.*` + heads | ~300 k |

Best checkpoint per strategy (selected among available iters: `5999/8999` for A, `11999/17999` for B and C).

## HCP probes

| Label | Metric | A (full FT) | B (freeze last3) | C (freeze fmri) |
|---|---|---|---|---|
| Sex | AUC | 0.751 | 0.696 | **0.844** |
| BrainVol | MAE | 116,179 | 117,744 | **99,455** |
| GrayMatterVol | MAE | 34,762 | 34,815 | **29,274** |
| Age | MAE | 3.044 | 3.056 | **2.931** |
| FluidIntel | MAE | 4.047 | 4.035 | **4.003** |
| ProcSpeed | MAE | 16.343 | 16.401 | **16.207** |
| WorkingMem | MAE | 10.442 | **10.410** | 10.411 |

## ADNI probes

| Label | Metric | A (full FT) | B (freeze last3) | C (freeze fmri) |
|---|---|---|---|---|
| Sex | AUC | 0.653 | 0.696 | **0.813** |
| CDR | AUC | **0.542** | 0.522 | 0.532 |
| Degradation1Y | AUC | **0.581** | 0.543 | 0.559 |
| Degradation2Y | AUC | **0.533** | 0.456 | **0.533** |
| Degradation3Y | AUC | 0.528 | 0.468 | **0.544** |
| MMSE | MAE | 2.478 | **2.426** | 2.750 |
| Age | MAE | 5.113 | **4.993** | 5.703 |

**Bold** = best value in row. AUC: higher is better. MAE: lower is better.

## Reproducibility

- Repo / branch: <https://github.com/DanAbergel/dinov2>, branch `fmri-mixed-t140`.
- YAML configs: `dinov2/configs/train/fmri_vits.yaml` (A), `fmri_vits_hcp_freeze_last3.yaml` (B), `fmri_vits_hcp_freeze_fmri.yaml` (C).
- Single sbatch: `slurm_jobs/run_dinov2_fmri.sh`, switch via `CONFIG_FILE=...`.
- Probes: `slurm_jobs/probe_adni.sh`, `slurm_jobs/probe_hcp.sh` (auto-launched by `slurm_jobs/probe_3ckpts.sh`).
- Aggregation: `python3 scripts/summarize_probes.py`.
