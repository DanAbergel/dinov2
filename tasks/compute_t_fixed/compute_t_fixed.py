"""Compute the maximal T_fixed for the mixed corpus.

Reads every scan's native T, upsamples it (round(T_native * tr / TARGET_TR)),
and reports the GLOBAL minimum = the largest T_fixed window that fits EVERY
scan with no padding. Also prints per-dataset minima and the shortest scan
(the one that caps T_fixed), so we can decide whether to drop outliers.

Usage:
    python compute_t_fixed.py [--margin 0]
"""

import argparse

from dinov2.data.fmri_data import compute_t_fixed_max, TARGET_TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=0,
                    help="Subtract this many frames from the min (safety margin)")
    args = ap.parse_args()

    t_max, per, argmin = compute_t_fixed_max(margin=args.margin)

    print("=" * 60)
    print(f"  TARGET_TR = {TARGET_TR}s   margin = {args.margin}")
    print("=" * 60)
    print(f"  T_fixed_max (global min upsampled T) = {t_max}")
    print(f"  -> window = {t_max * TARGET_TR:.1f}s of brain activity\n")
    print("  per-dataset min upsampled T:")
    for d, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"    {d:6s} {v}")
    print(f"\n  shortest scan (caps T_fixed):")
    print(f"    dataset = {argmin['dataset']}")
    print(f"    file    = {argmin['path'].split('/')[-1]}")
    print(f"    tr      = {argmin['tr']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
