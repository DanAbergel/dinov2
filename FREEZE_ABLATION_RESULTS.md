# Freeze ablation -- results

Three runs sharing the same setup (HCP-only, T=1200, temporal_kernel=20, base_lr=3e-4, effective batch 16, 20 epochs, DINOv2 ViT-S/14 reg4 ImageNet init). Only the parameter-freezing policy differs.

## Architecture: what we added on top of DINOv2

We keep the DINOv2 SSL algorithm (student-teacher multi-crop self-distillation with iBOT masked-patch prediction) unchanged. Four fMRI-specific pieces are added on top of the official code:

1. **3D+1D patchify** (`PatchEmbed3DPlus1D`). Replaces DINOv2's 2D `Conv2d` patch embed. A hierarchical encoder built from `Conv3Plus1d` blocks (separable spatial-3D + temporal-1D conv, in the MovieGen-TAE style). Input `(T, X, Y, Z)` -> output token grid `(T_eff = T / temporal_kernel, N_spatial = N_x * N_y * N_z)`. For HCP at T=1200 with `temporal_kernel=20` and `(45, 54, 45)` voxels at `patch_size=9`: yields `60 * 150 = 9000` patch tokens, plus 1 CLS + 4 register tokens = 9005 tokens fed to the ViT.

2. **Factorised positional embedding** (`PositionEmbedding3D`). Three small tables -- `pos_temporal` (T_eff), `pos_spatial` (N_spatial), `pos_cls` (1) -- broadcast and summed. `O(T_eff + N_spatial)` parameters instead of `O(T_eff * N_spatial)` for a flat 2D ViT pos table.

3. **3D multi-crop augmentation** (`MultiCrop3D`). For each scan, picks 2 global + 8 local random 3D spatial sub-volumes (scale `[0.32, 1.0]` and `[0.05, 0.32]` of the volume) and trilinearly resizes each back to `(X, Y, Z)`. No photometric augmentation (color jitter, blur, solarize, ImageNet normalize) and no flip (preserves L/R brain asymmetry) -- those don't apply to z-scored fMRI volumes.

4. **fMRI datasets** (`HCPFullScanDataset`, `ADNIFullScanDataset`, `MixedFMRIDataset`). Each scan loaded as `(T, 1, X, Y, Z)` with per-frame z-score. HCP scans are reconstructed from saved windowed `.pt` files; ADNI scans are slices of a shared 5D tensor.

Everything else (the SSL training loop, EMA teacher, DINO + iBOT + KoLeo losses, FSDP wrapping, sinkhorn-knopp centering, schedulers) is the official DINOv2 code, **unchanged**.

## Strategies and trainable parameter counts

Backbone counts come from the actual training logs (`backbone trainable=...` line printed by `apply_freeze_policy`). The DINO head is a 3-layer MLP with 4096 prototypes and 256 bottleneck (~6.57 M params, computed analytically); it is random-init and **always trainable**. iBOT shares the DINO head (`ibot.separate_head: false`), no extra head.

| Strategy | Backbone trained | Backbone trainable | Backbone frozen | DINO head | **TOTAL trainable** |
|---|---|---:|---:|---:|---:|
| **A -- Full fine-tune** | all blocks + patch_embed + norm | 36,212,256 | 0 | 6,566,144 | **42,778,400** |
| **B -- Freeze last 3** | patch_embed + blocks.9-11 + norm | 16,776,480 | 19,435,776 | 6,566,144 | **23,342,624** |
| **C -- Freeze except input** | patch_embed only | 11,450,016 | 24,762,240 | 6,566,144 | **18,016,160** |

The backbone contains the `PatchEmbed3DPlus1D` we added (~11.45 M -- much bigger than a 2D ViT patch embed because of the temporal `Conv3Plus1d` ResBlocks), 12 transformer blocks (~21.2 M total), and a dead-weight 2D `pos_embed` allocated by the ViT parent class (~3.46 M, unused in fMRI mode but still allocated).

Best checkpoint per strategy (selected among available iters: `5999/8999` for A, `11999/17999` for B and C). Note on A: the original run (`dinov2_fmri_20260524_144327`) stopped at iter 10,410 instead of completing the 20 epochs (19,999 iters), so its results aren't strictly at the same training depth as B and C. A is being re-run to completion (`fmri_vits_hcp_baseline.yaml`).

## HCP probes

| Label | Metric | MLP baseline (step5) | A (full FT) | B (freeze last3) | C (freeze fmri) |
|---|---|---:|---:|---:|---:|
| Sex | AUC | 0.935 | 0.751 | 0.696 | **0.844** |
| BrainVol | MAE | - | 116,179 | 117,744 | **99,455** |
| GrayMatterVol | MAE | - | 34,762 | 34,815 | **29,274** |
| Age | MAE | 4.228 | 3.044 | 3.056 | **2.931** |
| FluidIntel | MAE | - | 4.047 | 4.035 | **4.003** |
| ProcSpeed | MAE | - | 16.343 | 16.401 | **16.207** |
| WorkingMem | MAE | - | 10.442 | **10.410** | 10.411 |

(The HCP step5 MLP baseline run only covered Sex and Age; dashes mean it wasn't re-run for the remaining labels.)

## ADNI probes

| Label | Metric | MLP baseline (step5) | A (full FT) | B (freeze last3) | C (freeze fmri) |
|---|---|---:|---:|---:|---:|
| Sex | AUC | 0.806 | 0.653 | 0.696 | **0.813** |
| CDR | AUC | **0.559** | 0.542 | 0.522 | 0.532 |
| Degradation1Y | AUC | 0.496 | **0.581** | 0.543 | 0.559 |
| Degradation2Y | AUC | 0.523 | **0.533** | 0.456 | **0.533** |
| Degradation3Y | AUC | 0.465 | 0.528 | 0.468 | **0.544** |
| MMSE | MAE | 4.492 | 2.478 | **2.426** | 2.750 |
| Age | MAE | 7.78 | 5.113 | **4.993** | 5.703 |

**Bold** = best value in the row. AUC: higher is better. MAE: lower is better. The MLP baseline column is the 3-layer MLP on raw flattened time-series (step5 in the FAIR repo, 5-fold CV; subject-aware `StratifiedGroupKFold` / `GroupKFold` for ADNI).

\footnotesize *Source data: `outputs/probes/probe_{adni,hcp}_<run_name>_iter<iter>.json` -- generated by `slurm_jobs/probe_{adni,hcp}.sh`, aggregated by `python3 scripts/summarize_probes.py`. Baselines from `FAIR/logs/step5_{hcp,adni}.out`.*
