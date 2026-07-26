"""Extract a DENSE pool of CONSECUTIVE axial slices per subject (for neighbor-slice local crops).

Unlike the spanning pool used for montages (which picks slices spread across the whole brain),
this keeps EVERY brain-containing slice in order, so z-1 / z+1 are true anatomical neighbors.

Output: an ImageFolder dense_pool/<class>/subject_<id>_z<NNN>.png  (z zero-padded -> sortable).

  python fmri2d/extract_dense_pool.py \
      --input '/sci/.../HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt' \
      --output $LAB/brain2d_dense --axis 2 --size 224
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image

from extract_slices import load_volume, to_uint8


def subject_id(path):
    m = re.search(r"subject_(\d+)", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="NIfTI/.pt file or glob (quote it, ** for recursive)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--class-name", default="HCP")
    ap.add_argument("--axis", type=int, default=2, help="0=sag 1=cor 2=axial (default)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--area-frac", type=float, default=0.2, help="keep slices whose brain area > frac*max")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input, recursive=True)) if any(c in args.input for c in "*?[") else [args.input]
    dst = os.path.join(args.output, args.class_name)
    os.makedirs(dst, exist_ok=True)
    print(f"{len(files)} scans -> dense consecutive slices in {dst}", flush=True)

    total = 0
    for k, f in enumerate(files):
        try:
            vol = load_volume(f, "mean")               # (X, Y, Z)
            if vol.ndim != 3:
                print(f"  SKIP {f}: ndim={vol.ndim}")
                continue
            thr = vol.mean()
            areas = np.array([(np.take(vol, i, axis=args.axis) > thr).sum() for i in range(vol.shape[args.axis])])
            keep = np.where(areas > args.area_frac * areas.max())[0]     # consecutive brain slices
            sid = subject_id(f)
            for i in keep:
                sl = np.rot90(np.take(vol, int(i), axis=args.axis))
                im = Image.fromarray(to_uint8(sl)).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
                im.save(os.path.join(dst, f"subject_{sid}_z{int(i):03d}.png"))
            total += len(keep)
            if k < 5 or (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(files)} subject_{sid}: {len(keep)} slices (z {keep.min()}..{keep.max()})", flush=True)
        except Exception as e:
            print(f"  SKIP {f}: {e}", flush=True)

    print(f"done: {total} slices -> {args.output}  (ImageFolder root = {args.output})", flush=True)


if __name__ == "__main__":
    main()
