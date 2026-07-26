"""Extract a TEMPORAL pool: for each subject, ONE fixed axial slice (the best slice on the
temporal mean), across N CONSECUTIVE timepoints. For neighbor-in-TIME SSL: local crops =
the SAME slice at nearby timepoints (same anatomy, different BOLD noise).

Output: dense_pool/<class>/subject_<id>_z<TTT>.png where TTT = timepoint index. Naming the time
axis "_z" lets SliceNeighborsFolder + NeighborSliceAugmentation neighbor over TIME with zero code
changes (it just sorts by the number and takes the neighbors). DINO_NEIGHBOR_STRIDE = time spacing.

  python fmri2d/extract_temporal_pool.py \
      --input '/.../HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt' \
      --output $LAB/brain2d_temporal --n-times 24 --start 100
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image

from extract_slices import load_4d, best_slice_index, to_uint8


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
    ap.add_argument("--n-times", type=int, default=24, help="consecutive timepoints per subject")
    ap.add_argument("--start", type=int, default=100, help="first timepoint (skip the unstable start)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input, recursive=True)) if any(c in args.input for c in "*?[") else [args.input]
    dst = os.path.join(args.output, args.class_name)
    os.makedirs(dst, exist_ok=True)
    print(f"{len(files)} scans -> {args.n_times} timepoints/subject at a fixed slice, into {dst}", flush=True)

    total = 0
    for k, f in enumerate(files):
        try:
            arr, t_first = load_4d(f)                     # .pt -> (T,X,Y,Z)
            if arr.ndim != 4:
                print(f"  SKIP {f}: not 4D (ndim={arr.ndim})")
                continue
            if not t_first:
                arr = np.moveaxis(arr, -1, 0)             # -> (T,X,Y,Z)
            T = arr.shape[0]
            zidx = best_slice_index(arr.mean(axis=0), args.axis)   # fixed slice from temporal mean
            start = min(args.start, max(0, T - args.n_times))
            times = range(start, min(start + args.n_times, T))
            sid = subject_id(f)
            for j, t in enumerate(times):
                sl = np.rot90(np.take(arr[t], zidx, axis=args.axis))     # fixed slice, timepoint t
                im = Image.fromarray(to_uint8(sl)).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
                im.save(os.path.join(dst, f"subject_{sid}_z{j:03d}.png"))  # _z<TIME> -> dataset neighbors over time
            total += len(times)
            if k < 5 or (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(files)} subject_{sid}: slice {zidx}, times {start}..{start + args.n_times - 1}", flush=True)
        except Exception as e:
            print(f"  SKIP {f}: {e}", flush=True)

    print(f"done: {total} images -> {args.output}  (ImageFolder root = {args.output})", flush=True)


if __name__ == "__main__":
    main()
