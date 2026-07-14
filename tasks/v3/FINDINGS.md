# Why the SSL pretraining loss stays flat — investigation (July 2026)

**One-line result:** the flat loss is **not a bug and not a collapse** — it is the expected
behaviour of DINO/iBOT when all views are the **same volume** (no augmentation diversity),
so the objective has no learning signal.

---

## Symptom

The DINOv2 SSL loss never drops below its initialisation (~10.3 total), even when
**overfitting a handful of scans** with every regulariser removed. Per-component it sits at
`dino_local ~6.2`, `dino_global ~2.08`, `ibot ~2.08` across **every** configuration tried.

## Tests run (v3 overfit harness — `tasks/v3/train/train.sh`)

All runs: unfrozen, `freeze_last_layer_epochs=0`, `base_lr=8e-3`, healthy warmup.

| Run | Setup | Result |
|---|---|---|
| overfit1 | 1 scan, sinkhorn centering | flat |
| overfit_ctr | 1 scan, **EMA** centering | flat |
| overfit_var | 16 distinct scans, EMA, **wd=0** | flat |
| overfit_diag | + grad-norm logging | flat |
| dinolocal_frozen | 4 scans, **FIXED_WINDOW** (byte-identical), EMA, wd=0 | **flat: `dino_local` 5.96 → 6.24 over 680 steps** |

## What we ruled out — with evidence

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Missing upstream DINOv2 commits | ❌ | up to date; only 1 unrelated commit missing (weights-URL loading) |
| Bug in our code | ❌ | `git diff upstream/main`: DINO/iBOT loss computation **identical** to official; 6D ViT branch mirrors the 2D path; masks aligned to token grid; config values standard |
| Gradients don't reach the encoder | ❌ | `gnorm_backbone` **> 0** (0.2–3.8) → encoder receives gradients |
| Collapse (degenerate tokens) | ❌ | `cls_std ≈ 1.4`, stable → encoder produces **diverse** representations |

Instrumentation used for this (grad norms + encoder-output std) was added temporarily and
then removed once the questions were answered.

## The decisive experiment

`dino_local_crops_loss` compares **student local crop** vs **teacher global crop** — and in our
pipeline both are the **same full-volume tensor** of the same subject (within a sample, local
and global crops come from the same loaded window). With `FIXED_WINDOW=1` the window is pinned,
so the input is **byte-identical** every step.

```
dino_local:  step 0 → 5.96    step 220 → 6.25    step 680 → 6.24   (flat, even slightly up)
```

**On byte-identical views, with no weight decay and everything trainable, the DINO loss does
not descend.** This is the direct empirical demonstration of the conclusion.

## Conclusion

- The code and the encoder are **healthy**.
- DINO/iBOT reduce their loss by **aligning two DIFFERENT augmented views** of the same content
  (learning invariance). We removed all augmentation (masking-only, identical full-volume views,
  meeting §2), so there is **no gap to close** → no learning signal → the loss starts at its
  floor and stays there.
- The teacher is the student's own EMA: sharpening the student sharpens the target in lockstep
  ("chasing your own reflection"), so temperature alone cannot drive the loss down.
- This also explains the weak downstream probes: a normal-looking loss reached via a trivial
  task shapes no useful representation.

## Literature (the two established results this follows from)

- **SimCLR** (Chen et al., ICML 2020, arXiv:2002.05709): ablation — *"no single transformation
  suffices"*; augmentation **composition** is what creates the task. → zero augmentation = no task.
- **Wang & Isola** (ICML 2020, arXiv:2005.10242): SSL = **alignment** (pull together two
  *augmented* views) + uniformity. Identical views make alignment trivial → no signal.

No paper runs "DINO with identical views" literally — it is the corollary of the above, which our
`dinolocal_frozen` run demonstrates empirically.

## Fix direction (proven necessary, not just hypothesised)

Reintroduce a real gap between the student's and teacher's views, staying within DINO/iBOT:

1. **Different temporal windows per view** — the fMRI analogue of DINO's spatial crops (teacher
   sees window A, student window B of the same scan → "same brain, different time → same rep").
   Also gives the local crops a real meaning.
2. **3D augmentations** — noise, intensity scaling, anatomical flips.
3. Raise `patch_embed_lr_mult` (the randomly-initialised 3D encoder currently learns at LR×0.2).

## Debug tooling left in place (env-gated, no effect on normal runs)

- `FMRI_OVERFIT_N=k` (fmri_data.py) — train on only k HCP scans; prints their paths.
- `FMRI_FIXED_WINDOW=1` (fmri_data.py) — pin the temporal window start to 0 (byte-identical input).
- `tasks/v3/train/train.sh` — diagnostic training script (no auto-probe); flags `OVERFIT_N`,
  `FIXED_WINDOW`, `UNFROZEN`, `OVERRIDES`.
