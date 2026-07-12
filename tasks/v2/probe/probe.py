"""Leakage-free 70:30 probe for a trained fMRI run — CLI entry point.

All logic lives in probelib/ (encoder / head / metrics / datasets / splits / run);
this file is only the command line.

    python probe.py --run-dir .../runs/v2/base --dataset ADNI --out out.json
    python probe.py --run-dir .../runs/v2/base --dataset UCLA --mlp 256,128 --out out.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make probelib importable

from probelib.datasets import REGISTRY
from probelib.encoder import load_teacher
from probelib.run import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--mlp", default=None, help="MLP hidden sizes, e.g. '256,128' (default: linear)")
    ap.add_argument("--checkpoint", default="model_final.rank_0.pth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mlp = tuple(int(w) for w in args.mlp.split(",")) if args.mlp else None
    teacher = load_teacher(args.run_dir, args.checkpoint)
    run(REGISTRY[args.dataset](), teacher, args.run_dir, args.checkpoint, mlp, args.out)


if __name__ == "__main__":
    main()
