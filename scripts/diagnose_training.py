"""Diagnose a DINOv2 training run from its training_metrics.json.

DINOv2 writes one JSON line per iteration to `training_metrics.json` in the
run directory. Each line is a dict with the LR, momentum, teacher_temp, and
the loss BREAKDOWN: dino_global_crops_loss, dino_local_crops_loss, ibot_loss,
koleo_loss, total_loss, plus running averages.

This script reads that file and tells you:
  1. Did total loss actually move? (start -> end delta)
  2. WHICH component moved? (per-component start/end + delta)
  3. Is the teacher collapsing? (teacher_temp evolution + dino head entropy
     proxy if present)
  4. Did LR / momentum schedules execute as planned?
  5. A simple per-component verdict ("LEARNING", "FLAT", "COLLAPSED").

Reference points for a healthy DINOv2 run (ImageNet, K=4096 prototypes):
  - chance-level CE = log(4096) ~ 8.29 per head
  - typical START total_loss ~ 14-15  (DINO + iBOT + small KoLeo)
  - typical END   total_loss ~ 3-5    (50-100 epochs)
  - delta ~ -10  units over the run

A delta of less than -2 over 20 epochs is a red flag.

Usage:
    python scripts/diagnose_training.py outputs/dinov2_fmri_hcp_baseline
    python scripts/diagnose_training.py outputs/dinov2_fmri_hcp_baseline \\
        --bin 500   # smooth with 500-iter rolling mean
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Healthy reference (DINOv2 paper, K=4096 prototypes per head).
CHANCE_CE = np.log(4096)     # ~ 8.29
HEALTHY_DELTA = -8.0          # typical drop over 50-100 epochs
FLAT_THRESHOLD = -1.0         # delta > -1 over a full run = essentially flat
COLLAPSE_TOL = 0.3            # within 0.3 of chance_ce = likely collapsed


def load_metrics(metrics_path):
    """Yield (iter_idx, record_dict) for each line in the JSONL file."""
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(r)
    return records


def first_last(records, key, n_first=200, n_last=200):
    """Return (mean of first n_first records, mean of last n_last) for `key`,
    skipping NaN. Used to characterise start vs end without single-iter noise."""
    vals = [r[key] for r in records if key in r and r[key] is not None
            and isinstance(r[key], (int, float)) and np.isfinite(r[key])]
    if not vals:
        return None, None, 0
    n = len(vals)
    start = float(np.mean(vals[: min(n_first, n // 4 + 1)]))
    end = float(np.mean(vals[-min(n_last, n // 4 + 1):]))
    return start, end, n


def verdict(component_name, start, end, chance_ref=CHANCE_CE):
    """Categorise a single component's behaviour."""
    if start is None:
        return "(no data)"
    delta = end - start
    near_chance_at_end = abs(end - chance_ref) < COLLAPSE_TOL
    near_chance_at_start = abs(start - chance_ref) < COLLAPSE_TOL
    tags = []
    if delta < -2:
        tags.append(f"LEARNING (delta {delta:+.2f})")
    elif delta < FLAT_THRESHOLD:
        tags.append(f"weak (delta {delta:+.2f})")
    else:
        tags.append(f"FLAT (delta {delta:+.2f})")
    if near_chance_at_end and not near_chance_at_start:
        tags.append("COLLAPSED at end")
    elif near_chance_at_end and near_chance_at_start:
        tags.append("never learned (at chance throughout)")
    return "  |  ".join(tags)


def report(run_dir, bin_size=200):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "training_metrics.json"
    if not metrics_path.exists():
        # Fallback: some DINOv2 setups write to training_metrics.json.<rank>
        # or to a `logs/` subdir.
        candidates = list(run_dir.glob("training_metrics*")) + \
                     list(run_dir.glob("**/training_metrics.json"))
        if candidates:
            metrics_path = candidates[0]
        else:
            print(f"No training_metrics.json under {run_dir}")
            return
    print(f"Reading {metrics_path}")

    records = load_metrics(metrics_path)
    if not records:
        print("(metrics file is empty)")
        return
    print(f"Loaded {len(records)} log records")
    print()

    # Use 'iteration' if present, else infer from order.
    iters = [r.get("iteration", i) for i, r in enumerate(records)]
    if iters[0] is not None and iters[-1] is not None:
        print(f"Iter range: {iters[0]} -> {iters[-1]}")

    # ----- Loss components ------------------------------------------------
    KEYS = [
        ("total_loss", "TOTAL"),
        ("dino_local_crops_loss", "DINO local"),
        ("dino_global_crops_loss", "DINO global"),
        ("ibot_loss", "iBOT"),
        ("koleo_loss", "KoLeo"),
    ]
    print()
    print(f"  {'component':<22} {'start':>10} {'end':>10} {'delta':>10}  verdict")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}  {'-'*40}")
    for key, name in KEYS:
        s, e, n = first_last(records, key, bin_size, bin_size)
        if s is None:
            print(f"  {name:<22}  (missing)")
            continue
        chance_ref = CHANCE_CE if "dino" in key or "ibot" in key else None
        v = verdict(name, s, e, chance_ref or CHANCE_CE)
        print(f"  {name:<22} {s:>10.3f} {e:>10.3f} {e - s:>+10.3f}  {v}")

    # ----- LR / momentum / teacher_temp schedules -------------------------
    print()
    print(f"  {'schedule':<22} {'start':>10} {'end':>10} {'delta':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for key, name in [("lr", "LR"),
                       ("wd", "Weight decay"),
                       ("mom", "EMA momentum"),
                       ("teacher_temp", "Teacher temp"),
                       ("last_layer_lr", "Last-layer LR")]:
        s, e, n = first_last(records, key, bin_size, bin_size)
        if s is None:
            continue
        print(f"  {name:<22} {s:>10.6f} {e:>10.6f} {e - s:>+10.6f}")

    # ----- High-level verdict ---------------------------------------------
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    s, e, _ = first_last(records, "total_loss", bin_size, bin_size)
    if s is None:
        print("  (no total_loss in metrics)")
        return
    delta = e - s
    if delta < -4:
        print(f"  Total loss dropped by {-delta:.2f} units -- HEALTHY learning.")
    elif delta < -2:
        print(f"  Total loss dropped by only {-delta:.2f} units -- WEAK learning.")
    else:
        print(f"  Total loss dropped by {-delta:.2f} units only -- ESSENTIALLY FLAT.")
        print("  ==> The SSL pre-training is the bottleneck, not the probe.")

    # Per-component diagnosis hint.
    sd, ed, _ = first_last(records, "dino_global_crops_loss", bin_size, bin_size)
    si, ei, _ = first_last(records, "ibot_loss", bin_size, bin_size)
    if sd is not None and abs(sd - ed) < 0.5:
        print("  - DINO global crops loss did not move: views too similar.")
        print("    -> Try adding temporal augmentation, larger crop diversity.")
    if si is not None and abs(si - ei) < 0.5:
        print("  - iBOT loss did not move: masked-patch prediction not learning.")
        print("    -> Possibly too many tokens (9k); reduce T or temporal_kernel.")
    if e is not None and abs(e - CHANCE_CE) < COLLAPSE_TOL:
        print(f"  - End total_loss ~ log(K) = {CHANCE_CE:.2f}: likely COLLAPSE")
        print("    (teacher output became uniform, student matches uniform).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir",
                    help="path to outputs/dinov2_fmri_<run>")
    ap.add_argument("--bin", type=int, default=200,
                    help="window size for start/end averaging (default 200)")
    args = ap.parse_args()
    report(args.run_dir, args.bin)


if __name__ == "__main__":
    main()
