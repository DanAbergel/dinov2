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


def load_volume(path):
    """Return a 3D volume (X, Y, Z) from a scan, temporal-averaged if 4D.

    Handles two on-disk formats:
      - .pt  torch tensor, shape (T, X, Y, Z) or (T, 1, X, Y, Z)  -> mean over T (axis 0)
      - .nii/.nii.gz NIfTI, shape (X, Y, Z) or (X, Y, Z, T)       -> mean over T (axis 3)
    """
    if path.endswith(".pt"):
        import torch
        t = torch.load(path, map_location="cpu", weights_only=True)
        arr = t.float().numpy()
        arr = np.squeeze(arr)              # drop channel dim -> (T, X, Y, Z)
        if arr.ndim == 4:
            arr = arr.mean(axis=0)         # .pt: T is the FIRST axis
        return arr
    else:
        import nibabel as nib
        arr = np.squeeze(nib.load(path).get_fdata())
        if arr.ndim == 4:
            arr = arr.mean(axis=3)         # NIfTI: T is the LAST axis
        return arr


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


def save_slice(vol3d, axis, out_dir, cls, stem, size):
    idx = best_slice_index(vol3d, axis)
    sl = np.take(vol3d, idx, axis=axis)
    sl = np.rot90(sl)  # upright-ish; flip/rotate here if your orientation looks off
    im = Image.fromarray(to_uint8(sl)).convert("RGB")
    if size:
        im = im.resize((size, size), Image.BICUBIC)
    dst = os.path.join(out_dir, cls)
    os.makedirs(dst, exist_ok=True)
    im.save(os.path.join(dst, stem + ".png"))
    return idx, sl.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="NIfTI file or glob (use quotes, ** for recursive)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--class-name", default="all", help="ImageFolder subdir (label). DINO ignores it for SSL.")
    ap.add_argument("--axis", type=int, default=2, help="slice axis: 0=sagittal 1=coronal 2=axial (default)")
    ap.add_argument("--size", type=int, default=0, help="resize to NxN (0 = native; DINOv2 resizes anyway)")
    ap.add_argument("--all-axes", action="store_true", help="inspection: save all 3 planes (axis0/1/2) per scan")
    args = ap.parse_args()

    is_glob = any(c in args.input for c in "*?[")
    files = sorted(glob.glob(args.input, recursive=True)) if is_glob else [args.input]
    print(f"{len(files)} scan(s) found")

    for i, f in enumerate(files):
        try:
            data = load_volume(f)
            if data.ndim != 3:
                print(f"  SKIP {f}: unexpected ndim={data.ndim}")
                continue
            stem = os.path.basename(f).split(".nii")[0]
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
