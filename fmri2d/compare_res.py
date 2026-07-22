"""Side-by-side LOW-res vs HIGH-res QA grid. For each subject present in BOTH folders,
paste [ low-res | high-res ] and stack N subjects. One image to eyeball the sharpness gain.

Run on the login node (just PIL):
  python fmri2d/compare_res.py --lowres $LAB/brain2d/HCP --hires $LAB/brain2d_hires_test/HCP \
      --out fmri2d/samples/compare_res.png --n 6
"""
import argparse
import glob
import os
import re

from PIL import Image, ImageDraw

_SUBJ = re.compile(r"subject_(\d+)")


def index_by_subject(folder):
    d = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.png"))):
        m = _SUBJ.search(os.path.basename(p))
        if m and m.group(1) not in d:
            d[m.group(1)] = p
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowres", required=True, help="low-res ImageFolder class dir (e.g. brain2d/HCP)")
    ap.add_argument("--hires", required=True, help="high-res ImageFolder class dir (e.g. brain2d_hires_test/HCP)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cell", type=int, default=256)
    args = ap.parse_args()

    lo, hi = index_by_subject(args.lowres), index_by_subject(args.hires)
    common = sorted(set(lo) & set(hi))[: args.n]
    if not common:
        raise SystemExit(f"no common subjects between {args.lowres} and {args.hires}")

    c, pad, lbl = args.cell, 6, 22
    W = c * 2 + pad * 3
    H = len(common) * (c + lbl + pad) + pad
    canvas = Image.new("RGB", (W, H), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 2), "left = OUR downsampled (45x54)   |   right = S3 full-res (91x109)", fill="yellow")
    for i, sid in enumerate(common):
        y = pad + lbl + i * (c + lbl + pad)
        draw.text((pad, y), f"subject {sid}", fill="white")
        for j, path in enumerate((lo[sid], hi[sid])):
            im = Image.open(path).convert("RGB").resize((c, c), Image.BICUBIC)
            canvas.paste(im, (pad + j * (c + pad), y + lbl))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}: {len(common)} subjects (low | high)")


if __name__ == "__main__":
    main()
