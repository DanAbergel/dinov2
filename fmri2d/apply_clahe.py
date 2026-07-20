"""Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to every PNG in a
brain2d-style folder (root/<class>/*.png) and write the result to a new folder.

CLAHE boosts LOCAL contrast (non-affine), so the smooth/homogeneous fMRI slices get
real structure back — unlike a global z-score, which is affine and leaves the relative
contrast unchanged. Used to test whether less homogeneous images let the DINO/CLS loss
actually drop.

Usage (in the torch venv):
    python fmri2d/apply_clahe.py --src $LAB/brain2d --dst $LAB/brain2d_clahe
    python fmri2d/apply_clahe.py --src $LAB/brain2d --dst $LAB/brain2d_clahe --clip 2.0 --tiles 8
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def build_clahe(clip, tiles):
    """Return a function uint8_gray -> uint8_gray, using cv2 if available, else skimage."""
    try:
        import cv2
        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        return lambda a: c.apply(a), "cv2 CLAHE"
    except Exception:
        from skimage import exposure  # clip_limit is a fraction (0-1) in skimage
        return (lambda a: (exposure.equalize_adapthist(a, kernel_size=a.shape[0] // tiles,
                                                        clip_limit=min(0.05, clip / 40.0)) * 255).astype(np.uint8),
                "skimage CLAHE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source root (root/<class>/*.png)")
    ap.add_argument("--dst", required=True, help="destination root")
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit (cv2 scale)")
    ap.add_argument("--tiles", type=int, default=8, help="CLAHE tile grid size (tiles x tiles)")
    args = ap.parse_args()

    apply_clahe, method = build_clahe(args.clip, args.tiles)
    print(f"method: {method}  clip={args.clip} tiles={args.tiles}", flush=True)

    pngs = sorted(glob.glob(os.path.join(args.src, "*", "*.png")))
    if not pngs:
        raise SystemExit(f"no PNGs under {args.src}/*/")
    print(f"{len(pngs)} images to process", flush=True)

    for i, p in enumerate(pngs):
        rel = os.path.relpath(p, args.src)           # <class>/<file>.png
        out_path = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        arr = np.array(Image.open(p).convert("L"))
        Image.fromarray(apply_clahe(arr)).save(out_path)
        if (i + 1) % 200 == 0 or i + 1 == len(pngs):
            print(f"  {i + 1}/{len(pngs)}", flush=True)

    print(f"done -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
