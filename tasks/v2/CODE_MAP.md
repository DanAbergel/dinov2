# Code map — where each piece lives

For every concept in the pipeline, the exact file and function/class that implements it.

## Training (self-supervised pretraining)

| Concept | Location (`file : symbol`) |
|---|---|
| Constants (paths, per-dataset TR, T_FIXED, holdout) | `dinov2/data/fmri_const.py` |
| Discover scans (glob) | `dinov2/data/fmri_data.py : build_corpus_entries` |
| Read the corpus manifest + filter | `dinov2/data/fmri_data.py : entries_from_manifest` |
| Subject-level holdout (no leakage) | `dinov2/data/fmri_data.py : entries_from_manifest` (via `_load_split_map`) |
| Build the offline manifest | `dinov2/data/fmri_data.py : write_corpus_manifest` |
| TR harmonization — native window | `dinov2/data/fmri_data.py : _native_window` |
| TR harmonization — polyphase resample → 270 @ 0.72s | `dinov2/data/fmri_data.py : _temporal_resample` |
| Window → (270,1,45,54,45), z-scored | `dinov2/data/fmri_data.py : _finalize` |
| **Per-frame spatial z-score** | `dinov2/data/fmri_data.py : _zscore_per_frame` |
| Training dataset (5 sources) | `dinov2/data/fmri_data.py : MixedFMRIDataset` |
| Per-batch dataset quota | `dinov2/data/samplers.py : ProportionalInfiniteSampler` |
| Masking-only augmentation | `dinov2/data/fmri_data.py : FullVolumeViews3D` |
| Patchify (3D spatial + 1D temporal) | `dinov2/layers/patch_embed_3d_plus_1d.py : PatchEmbed3DPlus1D` |
| Model build (student/teacher, ViT-S) | `dinov2/models/__init__.py : build_model_from_cfg` |
| Freeze policy + augmentation selection + loop | `dinov2/train/train.py : do_train` |
| Config (dims, patch, freeze, LR, quota) | `dinov2/configs/train/fmri_vits.yaml` |
| Launch a run (+ auto-probe) | `tasks/v2/train/train.sh` |

## Probe (evaluation — encoder frozen)

| Concept | Location (`file : symbol`) |
|---|---|
| CLI entry point | `tasks/v2/probe/probe.py : main` |
| Load frozen teacher | `tasks/v2/probe/probelib/encoder.py : load_teacher` |
| **CLS extraction + TR harmonization (probe)** | `tasks/v2/probe/probelib/encoder.py : cls_token` |
| Probe head (linear / MLP) | `tasks/v2/probe/probelib/head.py : Probe` |
| **Training loop (Adam + BCE + backprop)** | `tasks/v2/probe/probelib/head.py : train_probe` |
| Metrics (AUROC / Acc / F1) | `tasks/v2/probe/probelib/metrics.py` |
| Datasets + comparison tasks | `tasks/v2/probe/probelib/datasets.py` (classes + `REGISTRY`) |
| **70:30 leakage-free split** | `tasks/v2/probe/probelib/splits.py : split_masks` |
| Orchestration (embed → split → probe → JSON) | `tasks/v2/probe/probelib/run.py : run` |
| Results report (PDF) | `tasks/v2/probe/make_report.py` |
| Full-featured legacy probe (used by finetune) | `tasks/v2/probe/probe_legacy.py` |

## Data downloaders (each a standalone task dir)

| Dataset | Location |
|---|---|
| ADHD-200 (fcp-indi, public) | `tasks/data_prep/download_adhd200/` |
| COBRE (figshare, public) | `tasks/data_prep/download_cobre/` |
| UCLA CNP ds000030 (OpenNeuro, public) | `tasks/data_prep/download_ucla/` |
| HCP task-fMRI (AWS, requester-pays) | `tasks/data_prep/download_hcp_task/` |
| OASIS AD-Conversion labels (from CDR) | `tasks/data_prep/fetch_oasis_labels/` |

## Ablation runners (probe)

| Runner | Location |
|---|---|
| All runs × datasets | `tasks/v2/probe/run_all_probes.sh` |
| Aggregation + MLP ablations (one job) | `tasks/v2/probe/run_probe_ablations.sh` |
| One probe per SLURM job (parallel) | `tasks/v2/probe/probe_one.sh` |
| Fan-out MLP ablation (one job per arch) | `tasks/v2/probe/launch_mlp_ablation.sh` |

## Docs

| Doc | Location |
|---|---|
| Detailed architecture (training + probe) | `tasks/v2/ARCHITECTURE.md` / `.pdf` |
| Results (colored) | `tasks/v2/probe/RESULTS.md` / `.pdf` |
| This map | `tasks/v2/CODE_MAP.md` |
