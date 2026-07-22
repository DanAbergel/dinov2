"""Build MONTAGE images: one image = a grid of many slices of the SAME subject (like an MRI
contact sheet). Each montage shows the whole brain, so the batch images are distinguishable by
the SUBJECT (not by slice height) -> DINO can drop AND learn subject features, not position.

Reads a slice pool (brain2d_pool/HCP/subject_<id>_*.png), and for each subject writes M montages
of GxG slices (default 4x4=16) into an ImageFolder layout. Then train/probe with ablation.sh on it.

Run (torch_env or plain python with pillow):
  python fmri2d/make_montages.py --src $LAB/brain2d_pool --dst $LAB/brain2d_montage --grid 4 --per-subject 8
"""
import argparse
import glob
import os
import random
import re

from PIL import Image

_SUBJ = re.compile(r"subject_(\d+)")
_Z = re.compile(r"_z(\d+)")


def _z(p):
    m = _Z.search(os.path.basename(p))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="slice pool root (root/<class>/subject_*.png)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--class-name", default="HCP")
    ap.add_argument("--grid", type=int, default=4, help="GxG montage (4 -> 16 slices)")
    ap.add_argument("--size", type=int, default=224, help="final montage size (NxN)")
    ap.add_argument("--per-subject", type=int, default=8, help="how many montages per subject")
    args = ap.parse_args()

    g = args.grid
    n = g * g                                   # slices per montage
    cell = args.size // g                        # pixels per slice
    out_dir = os.path.join(args.dst, args.class_name)
    os.makedirs(out_dir, exist_ok=True)

    by_subject = {}
    for p in sorted(glob.glob(os.path.join(args.src, "*", "*.png"))):
        m = _SUBJ.search(os.path.basename(p))
        if m:
            by_subject.setdefault(m.group(1), []).append(p)
    if not by_subject:
        raise SystemExit(f"no subject_<id> PNGs under {args.src}")
    print(f"{len(by_subject)} subjects; {g}x{g}={n} slices/montage, {cell}px/cell, "
          f"{args.per_subject} montages/subject", flush=True)

    made = 0
    for si, (sid, paths) in enumerate(sorted(by_subject.items())):
        stem = re.sub(r"_t\d+_z\d+$|_z\d+$", "", os.path.splitext(os.path.basename(paths[0]))[0])
        for m in range(args.per_subject):
            picks = (random.sample(paths, n) if len(paths) >= n
                     else [random.choice(paths) for _ in range(n)])
            picks = sorted(picks, key=_z)                    # order top->bottom = anatomical sheet
            montage = Image.new("L", (args.size, args.size))
            for i, p in enumerate(picks):
                tile = Image.open(p).convert("L").resize((cell, cell), Image.BICUBIC)
                r, c = divmod(i, g)
                montage.paste(tile, (c * cell, r * cell))
            montage.save(os.path.join(out_dir, f"{stem}_m{m:02d}.png"))
            made += 1
        if si < 3 or (si + 1) % 100 == 0:
            print(f"  {si + 1}/{len(by_subject)} {sid}", flush=True)

    print(f"done: {made} montages -> {args.dst}  (ImageFolder root = {args.dst})", flush=True)


if __name__ == "__main__":
    main()
