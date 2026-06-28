"""Download ABIDE I (+II) preprocessed rs-fMRI and downsample to (45,54,45).

ABIDE = Autism Brain Imaging Data Exchange. We use the Preprocessed
Connectomes Project (PCP) outputs hosted on the public fcp-indi S3 bucket
(anonymous access, no credentials).

Pipeline / strategy choice (to match our other datasets' MINIMAL preprocessing):
  - pipeline = cpac
  - strategy = nofilt_noglobal   (NO band-pass filtering, NO global signal
    regression — closest to the minimal preprocessing of HCP/OASIS/AOMIC)
  - derivative = func_preproc    (the 4D preprocessed BOLD, already in MNI)

Each func_preproc file is already motion-corrected and registered to MNI,
so we only need spatial downsampling (no registration pipeline like ADNI).

Streams per subject:
  1. Download <SITE>_<SUBID>_func_preproc.nii.gz from s3 (~50-150 MB)
  2. Resample (X, Y, Z, T) -> (T, 45, 54, 45) via trilinear
  3. Save .pt, delete raw

Usage:
    python tasks/data_prep/download_abide/download_abide.py \\
        --output-dir /sci/labs/arieljaffe/dan.abergel1/ABIDE_data/downsampled \\
        --tmp-dir /tmp/abide_raw
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from botocore import UNSIGNED
from botocore.client import Config

BUCKET = "fcp-indi"
TARGET_SHAPE = (45, 54, 45)

# PCP output prefixes. ABIDE I and ABIDE II have slightly different layouts.
PREFIXES = {
    "abide1": "data/Projects/ABIDE_Initiative/Outputs/cpac/nofilt_noglobal/func_preproc/",
    "abide2": "data/Projects/ABIDE2/Outputs/cpac/nofilt_noglobal/func_preproc/",
}


def make_s3_client():
    """Anonymous S3 client (fcp-indi is a public bucket)."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_func_preproc_keys(s3, prefix: str) -> list:
    """List all func_preproc .nii.gz keys under a PCP prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("func_preproc.nii.gz"):
                keys.append(k)
    return sorted(keys)


def subject_from_key(key: str) -> str:
    """'..._func_preproc/NYU_0051091_func_preproc.nii.gz' -> 'NYU_0051091'."""
    name = Path(key).name
    return name.replace("_func_preproc.nii.gz", "")


def download_file(s3, key: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(Bucket=BUCKET, Key=key, Filename=str(dest))


def downsample_4d(nii_path: Path) -> torch.Tensor:
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"expected 4D, got {data.ndim}D {data.shape}")
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)   # (1, T, X, Y, Z)
    vol_ds = F.interpolate(vol, size=TARGET_SHAPE, mode="trilinear",
                           align_corners=False)
    return vol_ds.squeeze(0)                                         # (T, 45, 54, 45)


def process_dataset(s3, name: str, prefix: str, output_dir: Path,
                    tmp_dir: Path, limit):
    print(f"\n{'#'*60}\n#  {name.upper()}\n{'#'*60}")
    print(f"Listing s3://{BUCKET}/{prefix} ...")
    keys = list_func_preproc_keys(s3, prefix)
    print(f"Found {len(keys)} func_preproc files.")
    if limit:
        keys = keys[:limit]
        print(f"Limiting to first {limit}.")

    out_ds_dir = output_dir / name
    out_ds_dir.mkdir(parents=True, exist_ok=True)

    for i, key in enumerate(keys, 1):
        subj = subject_from_key(key)
        out_path = out_ds_dir / f"{subj}_downsampled.pt"
        if out_path.exists():
            print(f"  [{i}/{len(keys)}] {subj} (already done, skip)")
            continue
        print(f"  [{i}/{len(keys)}] {subj}")
        tmp_nii = tmp_dir / f"{name}_{subj}.nii.gz"
        try:
            download_file(s3, key, tmp_nii)
            ds = downsample_4d(tmp_nii)
            torch.save(ds, out_path)
            print(f"    saved shape={tuple(ds.shape)}")
        except Exception as e:
            print(f"    [error] {subj}: {e}")
        finally:
            if tmp_nii.exists():
                tmp_nii.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/abide_raw")
    ap.add_argument("--datasets", default="abide1",
                    help="Comma-separated: abide1, abide2, or both. "
                         "Default abide1 (abide2 layout less standard).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in requested:
        if d not in PREFIXES:
            print(f"ERROR: unknown dataset {d!r}. Choices: {list(PREFIXES)}",
                  file=sys.stderr)
            sys.exit(2)

    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target shape : {TARGET_SHAPE}")
    print(f"Output       : {output_dir}")
    print(f"Datasets     : {requested}")

    s3 = make_s3_client()
    for d in requested:
        process_dataset(s3, d, PREFIXES[d], output_dir, tmp_dir, args.limit)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
