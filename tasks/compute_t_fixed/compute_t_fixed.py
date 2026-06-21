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

    t_max, per, argmin, per_all = compute_t_fixed_max(margin=args.margin)
    total = sum(len(v) for v in per_all.values())

    print("=" * 64)
    print(f"  TARGET_TR = {TARGET_TR}s   margin = {args.margin}   total scans = {total}")
    print("=" * 64)
    print(f"  T_fixed_max (keep ALL, no padding) = {t_max}  "
          f"-> {t_max * TARGET_TR:.1f}s window\n")
    print("  per-dataset min upsampled T:")
    for d, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"    {d:6s} {v}")
    print(f"\n  shortest scan (caps T_fixed): {argmin['dataset']} / "
          f"{argmin['path'].split('/')[-1]} (tr={argmin['tr']})")

    # Trade-off table: for each candidate T_fixed, how many scans are dropped
    # (those whose upsampled T < T_fixed would need padding).
    print("\n  window vs scans dropped:")
    print(f"    {'T_fixed':>7} {'window':>7} {'kept':>6} {'dropped':>8}   per-dataset dropped")
    candidates = sorted(set([t_max, 150, 200, 250, 271, 300, 350, 400, 500]))
    for c in candidates:
        dropped = {d: sum(1 for v in lst if v < c) for d, lst in per_all.items()}
        n_drop = sum(dropped.values())
        nz = {d: n for d, n in sorted(dropped.items()) if n}
        print(f"    {c:>7} {c*TARGET_TR:>6.1f}s {total-n_drop:>6} {n_drop:>8}   {nz}")
    print("=" * 64)


if __name__ == "__main__":
    main()
