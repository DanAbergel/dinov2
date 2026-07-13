"""Build the corpus manifest once.

Scans every scan's native T across the five datasets and writes
    <lab_root>/corpus_manifest.csv : dataset,path,subject_id,tr,T_native,upsampled_T

MixedFMRIDataset reads this (instead of re-scanning ~4600 headers on every run)
and uses upsampled_T to drop too-short scans for a given T_fixed.

Usage:
    python build_corpus_manifest.py [--out /path/corpus_manifest.csv]
"""

import argparse

from dinov2.data.fmri_const import LAB_ROOT, DEFAULT_MANIFEST
from dinov2.data.fmri_offline import write_corpus_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=LAB_ROOT)
    ap.add_argument("--out", default=None,
                    help=f"Output CSV (default: <lab>/{DEFAULT_MANIFEST})")
    args = ap.parse_args()
    out = args.out or f"{args.lab}/{DEFAULT_MANIFEST}"
    path = write_corpus_manifest(out, lab_root=args.lab)
    print(f"Manifest written: {path}")


if __name__ == "__main__":
    main()
