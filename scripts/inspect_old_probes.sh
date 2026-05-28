#!/bin/bash
# =====================================================================
# Inspect ALL probe JSONs (current + backup dirs) to trace where each
# metric came from. Specifically helps locate the "Degradation1Y ~0.612"
# value and identify which run/iter/config produced it.
#
# No GPU, no SLURM needed — just reads JSON. Run on the gateway:
#     bash scripts/inspect_old_probes.sh
# =====================================================================

set -euo pipefail
cd /sci/labs/arieljaffe/dan.abergel1/repos/FAIR_official

python3 << 'PY'
import json, glob, os

# Scan current probes dir AND every backup subdir.
patterns = [
    'outputs/probes/probe_iter*.json',
    'outputs/probes/probe_hcp_iter*.json',
    'outputs/probes/old*/probe_iter*.json',
    'outputs/probes/old*/probe_hcp_iter*.json',
]
files = []
for p in patterns:
    files += glob.glob(p)
files = sorted(set(files))

if not files:
    print("No probe JSONs found anywhere under outputs/probes/.")
    raise SystemExit

print(f"Found {len(files)} probe JSON file(s).\n")

# Collect every classification AUC across all files, so we can find the
# global best per (dataset, label) and where it came from.
rows = []   # (dataset, label, auc, run, iter, filepath)
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  WARN: cannot parse {f}: {e}")
        continue
    cfg = d.get('config', {})
    dataset = cfg.get('dataset', '?')
    ckpt = cfg.get('checkpoint', '')
    run = ckpt.split('/')[1] if '/' in ckpt else '?'
    it = cfg.get('iteration', -1)
    for r in d.get('results', []):
        if r['is_classification'] and 'AUC' in r['metrics']:
            auc = r['metrics']['AUC']['mean']
            rows.append((dataset, r['label'], auc, run, it, f))

# 1) Full dump, grouped by file.
print("="*70)
print("  FULL DUMP (all classification AUCs, per file)")
print("="*70)
last_file = None
for dataset, label, auc, run, it, f in sorted(rows, key=lambda x: (x[5], x[1])):
    if f != last_file:
        print(f"\n{f}")
        print(f"  run={run}  dataset={dataset}  iter={it}")
        last_file = f
    print(f"    {label:<18} AUC {auc:.3f}")

# 2) Best per (dataset, label) and where it came from.
print("\n" + "="*70)
print("  BEST AUC per (dataset, label) and its source")
print("="*70)
best = {}
for dataset, label, auc, run, it, f in rows:
    key = (dataset, label)
    if key not in best or auc > best[key][0]:
        best[key] = (auc, run, it, f)
for (dataset, label), (auc, run, it, f) in sorted(best.items()):
    tail = run.split('_')[-1] if '_' in run else run
    print(f"  {dataset:<5} {label:<18} AUC {auc:.3f}   <- run {tail} @ iter {it}")

# 3) Specifically flag any Degradation1Y >= 0.58 (the ~0.612 we're chasing).
print("\n" + "="*70)
print("  Degradation1Y AUC >= 0.58 (chasing the ~0.612)")
print("="*70)
hits = [(auc, run, it, f) for ds, lbl, auc, run, it, f in rows
        if lbl == 'Degradation1Y' and auc >= 0.58]
if hits:
    for auc, run, it, f in sorted(hits, reverse=True):
        print(f"  AUC {auc:.3f}  run={run}  iter={it}")
        print(f"           file={f}")
else:
    print("  None found. The ~0.612 was likely a misremembered value")
    print("  or a single lucky CV fold; max Degradation1Y is shown above.")
PY
