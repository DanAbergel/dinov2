"""Contact sheet: a grid of N evenly-spaced sample PNGs from a folder, to eyeball
variance / whether images are flat (all-similar) or rich. Just PIL, run on the login node:

  python fmri2d/sheet.py --src $LAB/brain2d_hires/HCP --out fmri2d/samples/hires_sheet.png --n 16
"""
import argparse
import glob
import math
import os

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of PNGs (e.g. brain2d_hires/HCP)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cell", type=int, default=200)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.png")))
    if not files:
        raise SystemExit(f"no PNGs in {args.src}")
    step = max(1, len(files) // args.n)
    pick = files[::step][: args.n]
    g = math.ceil(math.sqrt(len(pick)))
    c, pad, lbl = args.cell, 4, 16
    W = g * c + (g + 1) * pad
    H = g * (c + lbl) + (g + 1) * pad + lbl
    canvas = Image.new("RGB", (W, H), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 2), f"{os.path.basename(args.src.rstrip('/'))}: {len(pick)} of {len(files)} images", fill="yellow")
    for i, p in enumerate(pick):
        r, col = divmod(i, g)
        x = pad + col * (c + pad)
        y = lbl + pad + r * (c + lbl + pad)
        im = Image.open(p).convert("RGB").resize((c, c), Image.BICUBIC)
        canvas.paste(im, (x, y + lbl))
        draw.text((x, y), os.path.basename(p)[:26], fill="white")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}: {len(pick)} images from {args.src}")


if __name__ == "__main__":
    main()
