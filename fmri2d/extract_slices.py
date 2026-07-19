"""Extract ONE representative 2D brain slice per fMRI scan -> PNG (for standard 2D DINOv2).

For each NIfTI scan (X, Y, Z [, T]):
  1. temporal mean if 4D  -> clean anatomy-like image (averages out noise),
  2. pick the slice (along the chosen axis) with the LARGEST brain cross-section
     -> guarantees "the brain is clearly visible",
  3. robust 1-99 percentile contrast, save as an RGB PNG.

Output is an ImageFolder layout (out/<class>/<scan>.png) so it plugs straight into
the Imagenette DINOv2 pipeline (imagenette_test) with dataset_path=ImageFolder:root=out.

Needs: nibabel, pillow, numpy (all in torch_env).

Examples
--------
# Inspect ONE scan across all 3 planes (to see which axis looks like a brain):
  python extract_slices.py --input /path/to/scan.nii.gz --output inspect --all-axes
# Batch a whole tree with the chosen axis (2 = axial by default):
  python extract_slices.py --input '/sci/.../**/*.nii.gz' --output brain2d --class-name all --axis 2 --size 224
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def load_volume(path, tnorm="mean"):
    """Return a 3D volume (X, Y, Z) from a scan, reduced over time if 4D.

    Formats: .pt torch tensor (T, X, Y, Z) [T first]; .nii/.nii.gz (X, Y, Z, T) [T last].
    Temporal reduction `tnorm` (per-voxel over time):
      - mean : temporal mean            -> anatomy-like image (default)
      - tsnr : mean / (std + eps)        -> per-voxel temporal SNR (z-score-style normalisation)
      - tstd : std                        -> temporal std map (BOLD fluctuation / functional)
    (Note: z-scoring a voxel's series then averaging over time = 0 everywhere, so the
     useful realisation of "per-voxel z-score over time" is tsnr = mean/std.)
    """
    if path.endswith(".pt"):
        import torch
        arr = np.squeeze(torch.load(path, map_location="cpu", weights_only=True).float().numpy())
        tax = 0                              # .pt: T is the FIRST axis
    else:
        import nibabel as nib
        arr = np.squeeze(nib.load(path).get_fdata())
        tax = arr.ndim - 1                   # NIfTI: T is the LAST axis
    if arr.ndim != 4:
        return arr                           # already a single 3D volume
    mu = arr.mean(axis=tax)
    if tnorm == "mean":
        return mu
    sd = arr.std(axis=tax)
    if tnorm == "tsnr":
        return mu / (sd + 1e-6)
    if tnorm == "tstd":
        return sd
    raise ValueError(f"unknown tnorm: {tnorm}")


def best_slice_index(vol3d, axis):
    """Index of the slice (along `axis`) with the most brain (largest above-mean area)."""
    thr = vol3d.mean()
    areas = [(np.take(vol3d, i, axis=axis) > thr).sum() for i in range(vol3d.shape[axis])]
    return int(np.argmax(areas))


def to_uint8(slice2d):
    s = slice2d.astype(np.float32)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)  # robust contrast
    s = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)
    return (s * 255).astype(np.uint8)


def _write_png(sl, out_dir, cls, stem, size):
    sl = np.rot90(sl)  # upright-ish; flip/rotate here if your orientation looks off
    im = Image.fromarray(to_uint8(sl)).convert("RGB")
    if size:
        im = im.resize((size, size), Image.BICUBIC)
    dst = os.path.join(out_dir, cls)
    os.makedirs(dst, exist_ok=True)
    im.save(os.path.join(dst, stem + ".png"))


def save_slice(vol3d, axis, out_dir, cls, stem, size):
    idx = best_slice_index(vol3d, axis)
    sl = np.take(vol3d, idx, axis=axis)
    _write_png(sl, out_dir, cls, stem, size)
    return idx, sl.shape


def load_4d(path):
    """Return (array, t_is_first_axis). .pt -> (T,X,Y,Z); NIfTI -> (X,Y,Z,T)."""
    if path.endswith(".pt"):
        import torch
        return np.squeeze(torch.load(path, map_location="cpu", weights_only=True).float().numpy()), True
    import nibabel as nib
    return np.squeeze(nib.load(path).get_fdata()), False


def frame_volumes(arr, t_first, n_frames):
    """List of 3D volumes at n_frames evenly-spaced timepoints."""
    T = arr.shape[0] if t_first else arr.shape[-1]
    idxs = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    return [(arr[t] if t_first else arr[..., t]) for t in idxs]


def scan_stem(path):
    """Unique, clean output name: <parent_dir>_<filename-without-ext>.
    Many cohorts store the subject in the PARENT dir and reuse the same filename
    (e.g. HCP: subject_766563/rfMRI_REST1_LR_downsampled.pt) -> the parent prefix
    keeps PNGs from overwriting each other."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii", ".pt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}_{base}" if parent else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="NIfTI file or glob (use quotes, ** for recursive)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--class-name", default="all", help="ImageFolder subdir (label). DINO ignores it for SSL.")
    ap.add_argument("--axis", type=int, default=2, help="slice axis: 0=sagittal 1=coronal 2=axial (default)")
    ap.add_argument("--size", type=int, default=0, help="resize to NxN (0 = native; DINOv2 resizes anyway)")
    ap.add_argument("--tnorm", default="mean", choices=["mean", "tsnr", "tstd"],
                    help="temporal reduction: mean (anatomy) | tsnr=mean/std (z-score norm) | tstd=std")
    ap.add_argument("--n-frames", type=int, default=0,
                    help="if >0: save this many 2D axial slices at evenly-spaced TIMEPOINTS per scan "
                         "(same axial slice index) instead of a single temporal-reduced image")
    ap.add_argument("--all-axes", action="store_true", help="inspection: save all 3 planes (axis0/1/2) per scan")
    args = ap.parse_args()

    is_glob = any(c in args.input for c in "*?[")
    files = sorted(glob.glob(args.input, recursive=True)) if is_glob else [args.input]
    print(f"{len(files)} scan(s) found")

    for i, f in enumerate(files):
        try:
            stem = scan_stem(f)
            # multi-timepoint mode: K axial slices (same slice index) at K timepoints
            if args.n_frames > 0:
                arr, t_first = load_4d(f)
                if arr.ndim != 4:
                    save_slice(arr, args.axis, args.output, args.class_name, f"{stem}_t00", args.size)
                    continue
                zidx = best_slice_index(arr.mean(axis=(0 if t_first else -1)), args.axis)
                vols = frame_volumes(arr, t_first, args.n_frames)
                for j, vt in enumerate(vols):
                    _write_png(np.take(vt, zidx, axis=args.axis), args.output,
                               args.class_name, f"{stem}_t{j:02d}", args.size)
                if i < 5 or i % 200 == 0:
                    print(f"  {stem}: {len(vols)} frames, axial slice {zidx}")
                continue

            data = load_volume(f, args.tnorm)
            if data.ndim != 3:
                print(f"  SKIP {f}: unexpected ndim={data.ndim}")
                continue
            if args.all_axes:
                for ax in (0, 1, 2):
                    idx, shp = save_slice(data, ax, args.output, args.class_name, f"{stem}_axis{ax}", args.size)
                    print(f"  {stem}  axis{ax}: slice {idx}  {shp}")
            else:
                idx, shp = save_slice(data, args.axis, args.output, args.class_name, stem, args.size)
                if i < 5 or i % 200 == 0:
                    print(f"  {stem}: axis{args.axis} slice {idx}  {shp}")
        except Exception as e:
            print(f"  SKIP {f}: {e}")

    print(f"-> {args.output}/{args.class_name}/  (ImageFolder root = {args.output})")


if __name__ == "__main__":
    main()
