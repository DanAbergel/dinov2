"""Build a SUBJECT-LEVEL train/val/test split for the corpus.

Why subject-level (not scan-level): ADNI has ~3.78 scans/subject, so a scan-level
split would put the same subject in both pretrain and test -> leakage. We split
SUBJECTS, so all scans of a held-out subject stay out of pretraining.

For each dataset we shuffle its unique subjects with a fixed seed and split
70/15/15 (train/val/test). MixedFMRIDataset then excludes the val+test subjects
of the downstream datasets (ADNI/ABIDE/OASIS) from SSL pretraining, and the
downstream probes evaluate on the held-out test subjects — so the encoder never
saw them. This matches Brain-JEPA / SLIM-Brain (test held out of pretraining) and
makes results comparable to those SOTA benchmarks.

Reads:  <lab>/corpus_manifest.csv  (dataset, path, subject_id, ...)
Writes: <lab>/subject_split.json
    {seed, fractions, datasets: {DS: {train:[subj...], val:[...], test:[...]}}}

Usage:
    python make_subject_split.py [--seed 42] [--train 0.7 --val 0.15 --test 0.15]
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="/sci/labs/arieljaffe/dan.abergel1")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    manifest = Path(args.manifest or f"{args.lab}/corpus_manifest.csv")
    out = Path(args.out or f"{args.lab}/subject_split.json")
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "fractions must sum to 1"

    # dataset -> set of unique subjects
    subs = defaultdict(set)
    with open(manifest) as f:
        for row in csv.DictReader(f):
            subs[row["dataset"]].add(row["subject_id"])

    result = {"seed": args.seed,
              "fractions": {"train": args.train, "val": args.val, "test": args.test},
              "datasets": {}}
    print(f"{'dataset':8} {'subjects':>8} {'train':>6} {'val':>5} {'test':>5}")
    for ds in sorted(subs):
        s = sorted(subs[ds])                          # deterministic base order
        random.Random(args.seed).shuffle(s)           # seeded shuffle
        n = len(s)
        n_tr = int(round(n * args.train))
        n_va = int(round(n * args.val))
        train, val, test = s[:n_tr], s[n_tr:n_tr + n_va], s[n_tr + n_va:]
        result["datasets"][ds] = {"train": train, "val": val, "test": test}
        print(f"{ds:8} {n:>8} {len(train):>6} {len(val):>5} {len(test):>5}")

    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
