"""Multi-slice TEMPORAL pool: for each subject, M spanning axial slices, each across N consecutive
timepoints. For neighbor-in-time SSL with MANY slices per subject.

Use with DINO_NEIGHBOR_MODE=slice_time: one sample = (subject, slice, timepoint); its 8 local crops
= the SAME slice at nearby timepoints (same anatomy, different BOLD noise). More data + anatomical
diversity in the batch, while keeping the temporal-invariance objective per slice.

Output: root/<class>/subject_<id>_z<ZZZ>_t<TTT>.png  (ZZZ = slice index 0..M-1, TTT = timepoint 0..N-1)

  python fmri2d/extract_multislice_temporal.py \
      --input '/.../HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt' \
      --output $LAB/brain2d_ms_temporal --n-slices 10 --n-times 16 --start 100
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image

from extract_slices import load_4d, to_uint8


def subject_id(path):
    m = re.search(r"subject_(\d+)", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".pt/NIfTI file or glob (quote it)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--class-name", default="HCP")
    ap.add_argument("--axis", type=int, default=2, help="slice axis (2=axial)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-slices", type=int, default=10, help="spanning axial slices per subject")
    ap.add_argument("--n-times", type=int, default=16, help="consecutive timepoints per slice")
    ap.add_argument("--start", type=int, default=100, help="first timepoint (skip unstable start)")
    ap.add_argument("--area-frac", type=float, default=0.2, help="brain-slice threshold: area > frac*max")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input, recursive=True)) if any(c in args.input for c in "*?[") else [args.input]
    dst = os.path.join(args.output, args.class_name)
    os.makedirs(dst, exist_ok=True)
    print(f"{len(files)} scans -> {args.n_slices} slices x {args.n_times} timepoints/subject into {dst}", flush=True)

    total = 0
    for k, f in enumerate(files):
        try:
            arr, t_first = load_4d(f)                         # .pt -> (T,X,Y,Z)
            if arr.ndim != 4:
                print(f"  SKIP {f}: not 4D (ndim={arr.ndim})")
                continue
            if not t_first:
                arr = np.moveaxis(arr, -1, 0)                 # -> (T,X,Y,Z)
            T = arr.shape[0]
            zmean = arr.mean(axis=0)                          # (X,Y,Z)
            thr = zmean.mean()
            areas = np.array([(np.take(zmean, i, axis=args.axis) > thr).sum() for i in range(zmean.shape[args.axis])])
            valid = np.where(areas > args.area_frac * areas.max())[0]
            if len(valid) == 0:
                valid = np.arange(zmean.shape[args.axis])
            slices = np.unique(np.linspace(valid[0], valid[-1], args.n_slices).astype(int))
            start = min(args.start, max(0, T - args.n_times))
            times = range(start, min(start + args.n_times, T))
            sid = subject_id(f)
            for zi, z in enumerate(slices):
                for ti, t in enumerate(times):
                    sl = np.rot90(np.take(arr[t], int(z), axis=args.axis))
                    im = Image.fromarray(to_uint8(sl)).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
                    im.save(os.path.join(dst, f"subject_{sid}_z{zi:03d}_t{ti:03d}.png"))
                    total += 1
            if k < 5 or (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(files)} subject_{sid}: {len(slices)} slices x {len(times)} times", flush=True)
        except Exception as e:
            print(f"  SKIP {f}: {e}", flush=True)

    print(f"done: {total} images -> {args.output}  (ImageFolder root = {args.output})", flush=True)


if __name__ == "__main__":
    main()
