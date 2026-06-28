"""Inspect the multi-source fMRI corpus before wiring MixedFMRIDataset.

Prints, per dataset (HCP / ABIDE / OASIS-3 / AOMIC / ADNI):
  - file count + a sample path
  - tensor shape & dtype of a few sample scans (confirm (T,45,54,45) everywhere)
  - the per-scan T distribution (min/median/max)
  - where the native TR comes from:
      HCP   -> fixed 0.72s
      ABIDE -> per-site, parsed from the <SITE>_<id> filename (prints site counts)
      OASIS -> per-scan TR manifest CSV (prints path + head)
      AOMIC -> PIOP1 (0.75s) vs PIOP2 (2.0s): prints the dir layout so we can
               tell how the two protocols are distinguished on disk
      ADNI  -> per-scan TR manifest CSV (prints path + head)

Read-only. Run on the cluster (needs torch + the data). One-shot, CPU-only.

Usage:
    python inspect_corpus.py [--lab /sci/labs/arieljaffe/dan.abergel1]
"""

import argparse
import collections
import glob
import os
import statistics

import torch


def load_shape(path):
    try:
        t = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return tuple(t.shape), str(t.dtype)
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}", ""


def T_of(path):
    try:
        t = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return int(t.shape[0])
    except Exception:  # noqa: BLE001
        return None


def t_distribution(paths, sample=60):
    """Median/min/max of the T (first) axis over a sample of scans."""
    step = max(1, len(paths) // sample)
    Ts = [T_of(p) for p in paths[::step]]
    Ts = [t for t in Ts if t]
    if not Ts:
        return "n/a"
    return f"min={min(Ts)} median={int(statistics.median(Ts))} max={max(Ts)} (n_sampled={len(Ts)})"


def header(name):
    print("\n" + "=" * 64)
    print(f"  {name}")
    print("=" * 64)


def sample_shapes(paths, k=3):
    for p in paths[:k]:
        sh, dt = load_shape(p)
        print(f"    {os.path.basename(p):45s} -> shape={sh} {dt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="/sci/labs/arieljaffe/dan.abergel1")
    args = ap.parse_args()
    LAB = args.lab

    # ---------------- HCP ----------------
    header("HCP  (expected TR fixed 0.72s)")
    hcp = sorted(glob.glob(f"{LAB}/HCP_data/downsampled/subject_*/*.pt"))
    print(f"  files: {len(hcp)}")
    if hcp:
        print(f"  sample path: {hcp[0].replace(LAB, '')}")
        sample_shapes(hcp)
        print(f"  T dist: {t_distribution(hcp)}")

    # ---------------- ABIDE ----------------
    header("ABIDE  (TR per site -> parsed from <SITE>_<id> filename)")
    ab = sorted(glob.glob(f"{LAB}/ABIDE_data/downsampled/**/*.pt", recursive=True))
    print(f"  files: {len(ab)}")
    if ab:
        print(f"  sample path: {ab[0].replace(LAB, '')}")
        sample_shapes(ab)
        sites = collections.Counter(os.path.basename(f).split("_")[0] for f in ab)
        print(f"  sites ({len(sites)}): {dict(sorted(sites.items()))}")
        print(f"  T dist: {t_distribution(ab)}")

    # ---------------- OASIS-3 ----------------
    header("OASIS-3  (TR per scan -> manifest CSV, no fixed TR)")
    oa = sorted(glob.glob(f"{LAB}/OASIS3_data/downsampled/**/*.pt", recursive=True))
    print(f"  files: {len(oa)}")
    if oa:
        print(f"  sample path: {oa[0].replace(LAB, '')}")
        sample_shapes(oa)
        print(f"  T dist: {t_distribution(oa)}")
    mans = glob.glob(f"{LAB}/OASIS3_data/downsampled/*manifest*.csv")
    print(f"  manifests: {[m.replace(LAB, '') for m in mans]}")
    for m in mans[:2]:
        print(f"  head {os.path.basename(m)}:")
        for line in open(m).read().splitlines()[:4]:
            print(f"      {line}")

    # ---------------- AOMIC ----------------
    header("AOMIC  (PIOP1=0.75s vs PIOP2=2.0s -> how distinguished on disk?)")
    ao = sorted(glob.glob(f"{LAB}/AOMIC_data/**/*.pt", recursive=True))
    print(f"  files: {len(ao)}")
    if ao:
        # Show the relative paths so the PIOP1/PIOP2 split structure is visible.
        print("  sample relative paths (look for piop1/piop2 markers):")
        for f in ao[:6]:
            print(f"    {f.replace(LAB, '')}")
        sample_shapes(ao)
        print(f"  T dist: {t_distribution(ao)}")
        # Any 'piop' token anywhere in the paths?
        toks = collections.Counter()
        for f in ao:
            low = f.lower()
            for key in ("piop1", "piop2", "piop"):
                if key in low:
                    toks[key] += 1
        print(f"  piop tokens in paths: {dict(toks)}")
    # top-level AOMIC layout
    print("  top-level AOMIC_data/ entries:")
    for e in sorted(os.listdir(f"{LAB}/AOMIC_data"))[:20]:
        print(f"    {e}")

    # ---------------- ADNI ----------------
    header("ADNI  (TR per scan -> manifest CSV, default 3.0s)")
    ad = sorted(glob.glob(f"{LAB}/ADNI_data/downsampled/**/*.pt", recursive=True))
    print(f"  files: {len(ad)}")
    if ad:
        print(f"  sample path: {ad[0].replace(LAB, '')}")
        sample_shapes(ad)
        print(f"  T dist: {t_distribution(ad)}")
    adman = f"{LAB}/ADNI_data/downsampled/adni_manifest.csv"
    if os.path.exists(adman):
        print("  manifest head:")
        for line in open(adman).read().splitlines()[:3]:
            print(f"      {line}")

    # ---------------- TOTAL ----------------
    header("TOTAL")
    total = len(hcp) + len(ab) + len(oa) + len(ao) + len(ad)
    print(f"  HCP {len(hcp)} + ABIDE {len(ab)} + OASIS {len(oa)} + "
          f"AOMIC {len(ao)} + ADNI {len(ad)} = {total} scans")


if __name__ == "__main__":
    main()
