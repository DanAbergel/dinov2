"""probelib — the leakage-free fMRI probe, split into small single-purpose modules.

  metrics.py   evaluation metrics (auroc / accuracy / f1)
  encoder.py   frozen teacher + one CLS embedding per scan
  head.py      the probe head (linear / MLP) + its training loop
  datasets.py  one class per cohort (samples + comparison tasks) + REGISTRY
  splits.py    the 70:30 leakage-free train/test split
  run.py       orchestration: samples -> embeddings -> train head -> JSON
"""
