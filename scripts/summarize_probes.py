"""Aggregate + compare all linear-probe JSON results into tables.

Reads every outputs/probes/probe[_hcp]_iter*.json, groups by
(run, dataset, iteration), and prints one comparison table per dataset:
rows = labels, columns = (run, iter). The primary metric is AUC for
classification labels and MAE for regression labels. The best value per
row is marked with '*' (higher AUC / lower MAE).

No GPU, no SLURM — just reads JSON. Run on the gateway:
    python3 scripts/summarize_probes.py
    python3 scripts/summarize_probes.py --run dinov2_fmri_20260527_121536
    python3 scripts/summarize_probes.py --probes-dir outputs/probes
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def primary_metric(result):
    """Return (key, value, higher_is_better) for the headline metric."""
    m = result["metrics"]
    if result["is_classification"]:
        if "AUC" in m:
            return "AUC", m["AUC"]["mean"], m["AUC"]["std"], True
        return "Acc", m["Acc"]["mean"], m["Acc"]["std"], True
    return "MAE", m["MAE"]["mean"], m["MAE"]["std"], False


def load_all(probes_dir, run_filter=None):
    """-> {dataset: {label: {(run, iter): (key, mean, std, higher_better)}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    runs_seen = set()
    # Canonical linear probes only. probe_clinical_* and probe_nonlinear_* have
    # different output structures (AUPRC + linear-vs-MLP split) and are handled
    # by their own scripts; lump them here would crash primary_metric().
    for f in sorted(glob.glob(os.path.join(probes_dir, "*.json"))):
        base = os.path.basename(f)
        if not base.startswith("probe_"):
            continue
        if base.startswith("probe_clinical_") or base.startswith("probe_nonlinear_"):
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  WARN: could not parse {f}: {e}")
            continue
        cfg = d.get("config", {})
        dataset = cfg.get("dataset", "?")
        ckpt = cfg.get("checkpoint", "")
        run = ckpt.split("/")[1] if "/" in ckpt else "?"
        it = cfg.get("iteration", -1)
        if run_filter and run_filter not in run:
            continue
        runs_seen.add(run)
        for r in d.get("results", []):
            key, mean, std, hib = primary_metric(r)
            data[dataset][r["label"]][(run, it)] = (key, mean, std, hib)
    return data, sorted(runs_seen)


def print_dataset_table(dataset, label_map):
    # Collect all (run, iter) columns, sorted by run then iter.
    cols = sorted({c for row in label_map.values() for c in row})
    if not cols:
        return
    # Short column headers: <run_suffix>@<iter>
    def col_header(c):
        run, it = c
        # use the timestamp tail of the run name to keep it short
        tail = run.split("_")[-1] if "_" in run else run
        return f"{tail}@{it}"

    headers = [col_header(c) for c in cols]
    label_w = max(len(l) for l in label_map) + 1
    col_w = max(14, max(len(h) for h in headers) + 2)

    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    # Header row
    line = f"{'label':<{label_w}}" + "".join(f"{h:>{col_w}}" for h in headers)
    print(line)
    print("-" * len(line))

    for label in sorted(label_map):
        row = label_map[label]
        # Determine the best cell in this row.
        present = [(c, row[c]) for c in cols if c in row]
        if present:
            hib = present[0][1][3]
            best_val = (max if hib else min)(v[1] for _, v in present)
        cells = []
        for c in cols:
            if c not in row:
                cells.append(f"{'-':>{col_w}}")
                continue
            key, mean, std, hib = row[c]
            star = "*" if abs(mean - best_val) < 1e-9 else " "
            # Format: AUC as .3f, MAE could be large
            if key in ("AUC", "Acc"):
                txt = f"{mean:.3f}{star}"
            elif mean >= 1000:
                txt = f"{mean:.0f}{star}"
            else:
                txt = f"{mean:.3f}{star}"
            cells.append(f"{txt:>{col_w}}")
        # Append the metric kind to the label for clarity.
        kind = present[0][1][0] if present else "?"
        print(f"{label+' ('+kind+')':<{label_w}}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir", default="outputs/probes")
    ap.add_argument("--run", default=None,
                    help="Filter to runs whose name contains this string "
                         "(e.g. a timestamp). Default: all runs.")
    args = ap.parse_args()

    data, runs = load_all(args.probes_dir, args.run)
    if not data:
        print(f"No probe JSONs found under {args.probes_dir}"
              + (f" matching run '{args.run}'" if args.run else ""))
        return

    print(f"\nRuns found: {', '.join(runs)}")
    print("(* = best value in that row; higher AUC/Acc better, lower MAE better)")
    for dataset in sorted(data):
        print_dataset_table(dataset, data[dataset])
    print()


if __name__ == "__main__":
    main()
