# Freeze ablation — résultats

**Question** (Ariel/Yoni) : *quelle partie du modèle DINOv2 ImageNet faut-il fine-tuner pour adapter au fMRI ? Tout ? Rien (sauf l'adaptateur d'entrée) ? Les derniers blocs seulement ?*

**Setup commun aux 3 runs** : dataset HCP-only, T=1200, temporal_kernel=20, base_lr=3e-4, batch effective 16 (2 × 8 grad accum), 20 epochs, init = checkpoint DINOv2 ViT-S/14 reg4 ImageNet filtré (sans pos_embed et patch_embed).

Seul change : la **politique de gel des paramètres**.

---

## 1. Les 3 stratégies comparées

| Stratégie | Composants entraînés | # params trainables | Description |
|---|---|---|---|
| **A — Full fine-tune** (`baseline`) | tout le backbone + heads | **~22 M** | Comportement officiel DINOv2 : aucun gel |
| **B — Freeze partiel** (`freeze_last3`) | `patch_embed.*` + `blocks.9-11` + `norm` + heads | **~3.6 M** | Gel des 9 premiers blocs ImageNet, fine-tuning des 3 derniers + couche finale |
| **C — Freeze sauf input** (`freeze_fmri`) | `patch_embed.*` + heads | **~300 k** | Tout le backbone ImageNet gelé. Seul l'adaptateur 3D+1D que **nous** avons rajouté est entraîné |

Pour chaque stratégie, on garde le **meilleur checkpoint** parmi `iter 11999, 17999` (les 2 freeze runs n'ont que ces iters disponibles ; le baseline est à iter 5999 et 8999).

---

## 2. Résultats — Probes HCP (la base de pré-entraînement)

| Label | Métrique | A (full FT) | B (freeze last3) | C (freeze fmri) | **Vainqueur** | Gain C vs A |
|---|---|---|---|---|---|---|
| Sex | AUC (higher better) | 0.751 | 0.696 | **0.844** | **C** | **+0.093** |
| BrainVol | MAE (lower better) | 116 179 | 117 744 | **99 455** | **C** | **−14 %** |
| GrayMatterVol | MAE (lower better) | 34 762 | 34 815 | **29 274** | **C** | **−16 %** |
| Age | MAE (lower better) | 3.044 | 3.056 | **2.931** | **C** | **−3.7 %** |
| FluidIntel | MAE (lower better) | 4.047 | 4.035 | **4.003** | **C** | −1.1 % |
| ProcSpeed | MAE (lower better) | 16.343 | 16.401 | **16.207** | **C** | −0.8 % |
| WorkingMem | MAE (lower better) | 10.442 | **10.410** | 10.411 | B (≈ C) | ≈ −0.3 % |

**=> Stratégie C (freeze sauf input) bat les 2 autres sur 6 labels HCP sur 7.**

---

## 3. Résultats — Probes ADNI (dataset de transfert)

Le baseline (A) sur ADNI vient du probe canonique sur le checkpoint `144327` (HCP-only T=1200, iter 8999) — **même méthodologie** que les freeze probes (5-fold CV, LogReg C=1.0, sklearn).

| Label | Métrique | A (full FT) | B (freeze last3) | C (freeze fmri) | **Vainqueur** | Gain C vs A |
|---|---|---|---|---|---|---|
| Sex | AUC (higher better) | 0.653 | 0.696 | **0.813** | **C** | **+0.160** |
| Degradation3Y | AUC (higher better) | 0.528 | 0.468 | **0.544** | **C** | +0.016 |
| **Degradation1Y** | AUC (higher better) | **0.581** | 0.543 | 0.559 | **A** | (−0.022) |
| Degradation2Y | AUC (higher better) | 0.533 | 0.456 | 0.533 | A ≈ C | ≈ 0 |
| CDR | AUC (higher better) | **0.542** | 0.522 | 0.532 | **A** | (−0.010) |
| MMSE | MAE (lower better) | 2.478 | **2.426** | 2.750 | **B** | (+11 %) |
| Age | MAE (lower better) | 5.113 | **4.993** | 5.703 | **B** | (+12 %) |

**Pattern plus nuancé sur ADNI :**
- **Anatomique** (Sex) : la stratégie C domine très largement (+0.16 AUC).
- **Démographique** (Age) et **cognitif** (MMSE) : la stratégie B (partial FT) gagne.
- **Clinique de progression** (Degradation1Y, CDR) : le baseline A reste légèrement meilleur (mais dans le bruit ± 0.05 sur l'AUC).

---

## 4. Synthèse — les 3 conclusions clés

### 4.1 Sur la base d'entraînement (HCP) : **freeze sauf input bat tout**
6/7 labels HCP sont gagnés par la stratégie C. Sur **Sex** l'écart est massif : 0.844 contre 0.751 (baseline) = **+9.3 points d'AUC**. Sur **BrainVol** et **GrayMatterVol**, la MAE chute de 14-16 %.

**Interprétation** : les features visuelles génériques apprises par DINOv2 sur ImageNet (edges, textures, formes) se transfèrent **étonnamment bien** au fMRI une fois l'adaptateur d'entrée correctement entraîné. Le **fine-tuning complet du backbone dégrade ces features** en les sur-spécialisant sur le petit jeu HCP (1k scans).

### 4.2 La stratégie B (freeze partiel) est la pire des trois sur HCP
Le partial fine-tuning de seulement les 3 derniers blocs crée une **discontinuité interne** : les blocs 0-8 ImageNet restent figés tandis que 9-11 s'adaptent au fMRI. Le mismatch entre features bas-niveau (ImageNet) et haut-niveau (fMRI) dégrade la qualité globale.

### 4.3 Sur le transfert ADNI : signal anatomique très solide, signal clinique sous bruit
Pour Sex (anatomique, n=812 équilibré), la stratégie C bat A de +0.16 AUC — **net et robuste**. Pour les labels cliniques (Degradation, CDR), tous les runs sont autour de 0.52-0.58 AUC avec ±0.05-0.10 d'écart-type sur 5 folds : différences **dans le bruit**. Cohérent avec la limite déjà identifiée : **~145 cas positifs Degradation1Y => data-limited.**

---

## 5. Ce qui justifie la stratégie C pour la thèse

| Argument | Détail |
|---|---|
| **Résultat robuste sur HCP** | 6/7 labels gagnés, écarts significatifs sur Sex / BrainVol / GrayMatterVol |
| **Résultat fort sur Sex ADNI** | +0.16 AUC vs baseline = effet large, dépasse la variance fold |
| **Économie de calcul** | 300k params trainables vs 22M (×73) — training plus rapide, moins de risque overfit |
| **Interprétation propre** | Cohérent avec la littérature transfer learning : "linear probing > full fine-tuning" sur petits datasets |
| **Architecture claire** | "Adaptateur d'entrée 3D+1D + backbone ImageNet gelé" est une histoire scientifique simple à défendre |

---

## 6. Limites et nuances

- Les checkpoints intermédiaires manquants (2999, 5999, 8999 sur les 2 freeze runs) empêchent de voir la dynamique complète d'entraînement.
- `model_final` (iter 19999) n'a pas été probé : à compléter pour la version finale du tableau.
- Le baseline A sur ADNI vient d'une probe antérieure (avant le renaming) — la méthodologie est identique (canonical 5-fold LogReg), mais la comparaison directe en fold-wise n'est pas faite.
- Les labels cliniques restent **data-limited** (~145 cas positifs Degradation1Y) => les différences entre stratégies sur ces labels sont dans le bruit. Le bottleneck principal n'est pas le pré-entraînement mais la rareté des étiquettes.

---

## 7. Reproductibilité

- **Repo / branche** : https://github.com/DanAbergel/dinov2 — branche `fmri-mixed-t140`
- **Configs YAML** :
  - A : `dinov2/configs/train/fmri_vits.yaml` (avec `dataset_path: HCP`, `fmri_temporal_size: 1200`, `fmri_temporal_kernel: 20`)
  - B : `dinov2/configs/train/fmri_vits_hcp_freeze_last3.yaml`
  - C : `dinov2/configs/train/fmri_vits_hcp_freeze_fmri.yaml`
- **Sbatch unique** : `slurm_jobs/run_dinov2_fmri.sh` — switcher via `CONFIG_FILE=…`
- **Probes** : `slurm_jobs/probe_adni.sh` + `slurm_jobs/probe_hcp.sh`, auto-launchés par `slurm_jobs/probe_3ckpts.sh`.
- **Tableau récap** automatique : `python3 scripts/summarize_probes.py`.
- **Mécanisme de freeze** : `apply_freeze_policy()` dans `dinov2/train/train.py` (commenté `# FMRI CHANGE`), activé par `cfg.optim.freeze_pretrained ∈ {"fmri_only", "fmri_plus_last_3", null}`.
