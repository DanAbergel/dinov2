# Changements apportés à DINOv2 officiel pour le fMRI

**Fork** : https://github.com/DanAbergel/dinov2 — branche `fmri-mixed-t140`
**Upstream** : https://github.com/facebookresearch/dinov2

L'algorithme SSL officiel de DINOv2 (SSLMetaArch, DINO/iBOT/KoLeo losses, FSDP, EMA teacher) est **strictement inchangé**. On ajoute 3 nouveaux fichiers pour le côté fMRI et on fait des edits **chirurgicaux et conditionnels** sur 8 fichiers officiels. Chaque edit est marqué `# FMRI CHANGE: ... WHY: ...`.

> **Default-preserving** : si tu retires les flags fmri du config (`fmri_mode`, `fmri_augmentation`, `grad_accum_steps`), les branches conditionnelles retombent sur le comportement officiel.

---

## 1. Nouveaux fichiers (entièrement les nôtres)

| Fichier | Rôle |
|---|---|
| `dinov2/data/fmri_data.py` | Datasets HCP / ADNI / Mixed + `MultiCrop3D` (crops spatiaux 3D) |
| `dinov2/layers/patch_embed_3d_plus_1d.py` | `Conv3Plus1d`, `_ResBlock3Plus1d`, `PatchEmbed3DPlus1D`, `PositionEmbedding3D` |
| `dinov2/configs/train/fmri_vits.yaml` | Config SSL fMRI (overrides `ssl_default_config.yaml`) |

---

## 2. Fichiers officiels modifiés (8 fichiers, edits localisés)

| Fichier | Ce qu'on a ajouté | Pourquoi |
|---|---|---|
| `models/__init__.py` | `embed_layer=` kwarg sur `build_model` + branche `fmri_mode` dans `build_model_from_cfg` | passer notre `PatchEmbed3DPlus1D` au ViT sans forker la factory |
| `models/vision_transformer.py` | branche early-return pour input 6D dans `prepare_tokens_with_masks` | les volumes fMRI sont 6D `(B, T, C, X, Y, Z)` et utilisent un pos embed factorisé ; le chemin officiel 4D reste intact en dessous |
| `train/train.py` | (a) branche `fmri_augmentation` (MultiCrop3D + grille mask `(T_eff, N_spatial)`), (b) gradient accumulation, (c) `int()` casts sur les iter counts, (d) helper `optimizer_step_and_ema` | (a) crops 3D, (b) effective batch 16 depuis batch physique 2, (c) supporter `warmup_epochs` fractionnaire (ex 0.3), (d) garder la boucle `do_train` lisible avec grad accum |
| `train/ssl_meta_arch.py` | `loss_scale` param sur `forward_backward` (one-liner `loss/loss_scale`) + guard `hasattr(_streams)` | compagnon de la grad accum (scale loss par micro-step) + compat PyTorch ≥2.3 pour les internals FSDP |
| `utils/config.py` | `effective_batch` inclut `grad_accum_steps` dans le scaling sqrt LR | la règle de scaling LR doit connaître le vrai batch |
| `data/loaders.py` | 3 elifs enregistrant les dataset strings `"HCP"` / `"ADNI"` / `"Mixed"` | rendre les nouveaux datasets résolvables depuis le YAML |
| `data/__init__.py` | re-export `MultiCrop3D` | même surface d'import que `DataAugmentationDINO` |
| `data/datasets/__init__.py` | re-export `HCPFullScanDataset` / `ADNIFullScanDataset` / `MixedFMRIDataset` | idem, pour `_parse_dataset_str` |

**Total :** ~240 lignes de modifs dans les fichiers officiels.

---

## 3. Fichiers NON touchés (la vraie black box)

L'algorithme SSL lui-même est **entièrement intact** :
- `dinov2/loss/` — DINO, iBOT, KoLeo, distillation
- `dinov2/fsdp/` — sharding, all_gather
- `dinov2/distributed/`
- `dinov2/layers/{attention, block, mlp, layer_scale, drop_path}.py`
- `dinov2/data/{samplers, masking, collate, augmentations, transforms}.py`
- `dinov2/logging/helpers.py` — revenu à l'officiel après nettoyage

---

## 4. Comment retrouver chaque edit

Chaque modif dans un fichier officiel est marquée :
```
# FMRI CHANGE: <quoi>
# WHY: <raison>
```
Pour les lister toutes :
```bash
grep -rn "FMRI CHANGE" dinov2/
```

---

## 5. Schéma mental du pipeline fMRI

```
fmri_data.py            patch_embed_3d_plus_1d.py
  (Datasets HCP/ADNI)     (PatchEmbed3DPlus1D + PositionEmbedding3D)
        │                          │
        ▼                          ▼
   loaders.py             models/__init__.py (embed_layer=...)
   (HCP/ADNI/Mixed)              │
        │                          ▼
        └──► train.py ──► vision_transformer.py (branche 6D)
                  │                │
                  ▼                ▼
           SSLMetaArch ──► loss/{dino,ibot,koleo}.py
           (officiel)        (officiel — intact)
```

Les **datasets** et le **patchify 3D+1D + pos embed** sont écrits par nous ; les **edits sur le chemin officiel** sont juste là pour brancher ces composants dans `do_train` + le ViT + le builder.
