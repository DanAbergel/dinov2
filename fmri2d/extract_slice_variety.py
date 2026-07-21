"""Extract brain slices at VARIED positions, to break the inter-image similarity that
pins DINO's target to uniform (teacher_entropy_ratio = 1.0).

Two modes (both produce N images spanning N different slice POSITIONS, so the only
difference between them is the subject factor):

  --mode onesubj    : ONE subject, N slices evenly spanning its brain (position varies,
                      subject fixed). Isolates "does slice variety alone unfreeze DINO?".
  --mode multisubj  : N subjects, one slice each at an evenly-spaced position (subject i ->
                      position i). Position AND subject vary.

Positions are chosen by linspace over the valid slice range (slices whose brain area is
> 20% of the max), so it's deterministic and covers low->high cuts.

Output = ImageFolder layout (out/<class>/*.png). Run in torch_env.

Examples
--------
  python fmri2d/extract_slice_variety.py --mode onesubj   --n-slices 16 --axis 2 --size 224 \
      --input '$LAB/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt' --output $LAB/brain2d_1subj
  python fmri2d/extract_slice_variety.py --mode multisubj --n-slices 16 --axis 2 --size 224 \
      --input '$LAB/HCP_data/downsampled/subject_*/rfMRI_REST1_LR_downsampled.pt' --output $LAB/brain2d_16subj
"""
import argparse
import glob

import numpy as np

from extract_slices import load_volume, _write_png, scan_stem


def valid_slice_indices(vol3d, axis, frac=0.2):
    """Indices of slices (along axis) whose brain area is > frac * max area."""
    thr = vol3d.mean()
    areas = np.array([(np.take(vol3d, i, axis=axis) > thr).sum() for i in range(vol3d.shape[axis])])
    keep = np.where(areas > frac * areas.max())[0]
    return keep if len(keep) else np.arange(vol3d.shape[axis])


def spanning_positions(vol3d, axis, n):
    """n slice indices evenly spanning the valid (brain-bearing) range."""
    v = valid_slice_indices(vol3d, axis)
    return np.linspace(v[0], v[-1], n).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of scans (quote it, ** for recursive)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", required=True, choices=["onesubj", "multisubj"])
    ap.add_argument("--n-slices", type=int, default=16)
    ap.add_argument("--class-name", default="HCP")
    ap.add_argument("--axis", type=int, default=2, help="0=sagittal 1=coronal 2=axial(default)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--tnorm", default="mean", choices=["mean", "tsnr", "tstd"])
    args = ap.parse_args()

    files = sorted(glob.glob(args.input, recursive=True))
    if not files:
        raise SystemExit(f"no scans matched {args.input}")
    print(f"{len(files)} scan(s) found; mode={args.mode}")

    if args.mode == "onesubj":
        f = files[0]
        vol = load_volume(f, args.tnorm)
        stem = scan_stem(f)
        idxs = spanning_positions(vol, args.axis, args.n_slices)
        print(f"  one subject {stem}: slices {list(idxs)}")
        for idx in idxs:
            _write_png(np.take(vol, idx, axis=args.axis), args.output,
                       args.class_name, f"{stem}_z{idx:03d}", args.size)

    else:  # multisubj: subject i -> position i (evenly spaced across ITS OWN valid range)
        chosen = files[: args.n_slices]
        fracs = np.linspace(0.0, 1.0, len(chosen))            # 0=low cut ... 1=high cut
        for f, fr in zip(chosen, fracs):
            vol = load_volume(f, args.tnorm)
            v = valid_slice_indices(vol, args.axis)
            idx = int(v[int(round(fr * (len(v) - 1)))])       # position at fraction fr of valid range
            stem = scan_stem(f)
            _write_png(np.take(vol, idx, axis=args.axis), args.output,
                       args.class_name, f"{stem}_z{idx:03d}", args.size)
            print(f"  {stem}: slice {idx}")

    print(f"-> {args.output}/{args.class_name}/  (ImageFolder root = {args.output})")


if __name__ == "__main__":
    main()
