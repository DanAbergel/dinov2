"""Download AOMIC PIOP1+PIOP2 rs-fMRI from OpenNeuro and downsample to (45,54,45).

AOMIC = Amsterdam Open MRI Collection (Snoek et al., Scientific Data 2021).
  - PIOP1 = OpenNeuro dataset ds002785 (~216 subjects)
  - PIOP2 = OpenNeuro dataset ds002790 (~226 subjects)

Both are OPEN ACCESS — no DUA, no credentials, no requester-pays.
We hit the anonymous s3://openneuro.org/ bucket via boto3 with UNSIGNED config.

Streams one session at a time:
  1. Download the fMRIPrep-preprocessed MNI152NLin2009cAsym rest BOLD
     (path: derivatives/fmriprep/sub-XXXX/func/<file>)
  2. Resample to (45, 54, 45) via trilinear interpolation
  3. Save as .pt with shape (T, 45, 54, 45) float32
  4. Delete raw NIfTI

Usage:
    python tasks/data_prep/download_aomic/download_aomic.py \\
        --output-dir /sci/labs/arieljaffe/dan.abergel1/AOMIC_data/downsampled \\
        --tmp-dir /tmp/aomic_raw
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

BUCKET = "openneuro.org"
DATASETS = {
    "piop1": "ds002785",
    "piop2": "ds002790",
}
TARGET_SHAPE = (45, 54, 45)

# Match the fMRIPrep-preprocessed MNI rest BOLD. Tolerates different
# acquisition labels (acq-mbep2d / acq-mb3 / etc.) and the two MNI
# template flavors. AOMIC uses task-restingstate; some BIDS variants
# may use task-rest. We accept both.
PREPROC_PATTERN = re.compile(
    r"derivatives/fmriprep/sub-([^/]+)/func/"
    r"sub-\1_task-(rest|restingstate)[^/]*"
    r"space-MNI152NLin(2009cAsym|6Asym)"
    r"[^/]*desc-preproc_bold\.nii\.gz$"
)


def make_s3_client():
    """Anonymous S3 client (no credentials)."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_preproc_keys(s3, dataset_id: str) -> list:
    """Find all 'desc-preproc_bold' files matching our rest pattern."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=f"{dataset_id}/derivatives/fmriprep/",
    ):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if PREPROC_PATTERN.search(k):
                keys.append(k)
    return sorted(keys)


def subj_from_key(key: str) -> str:
    """Extract subject ID from a BIDS-style S3 key."""
    m = re.search(r"sub-([^/]+)/func/", key)
    return m.group(1) if m else "unknown"


def download_file(s3, key: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(Bucket=BUCKET, Key=key, Filename=str(dest))


def downsample_4d(nii_path: Path) -> torch.Tensor:
    """Load 4D NIfTI, downsample spatial to TARGET_SHAPE via trilinear."""
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)               # (X, Y, Z, T)
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)
    vol_ds = F.interpolate(
        vol,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return vol_ds.squeeze(0)                              # (T, 45, 54, 45)


def process_dataset(s3, dataset_key: str, dataset_id: str,
                    output_dir: Path, tmp_dir: Path, limit: int | None):
    """Process all subjects in one OpenNeuro dataset (PIOP1 or PIOP2)."""
    print(f"\n{'#'*60}")
    print(f"#  {dataset_key.upper()}  ({dataset_id})")
    print(f"{'#'*60}")
    print(f"Listing {dataset_id}/derivatives/fmriprep/ on s3://{BUCKET} ...")
    keys = list_preproc_keys(s3, dataset_id)
    print(f"Found {len(keys)} preprocessed rest BOLD files.")
    if limit:
        keys = keys[:limit]
        print(f"Limiting to first {limit}.")

    out_dataset_dir = output_dir / dataset_key
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    for i, key in enumerate(keys, 1):
        subj = subj_from_key(key)
        out_subj_dir = out_dataset_dir / f"sub-{subj}"
        out_path = out_subj_dir / "restingstate_downsampled.pt"
        if out_path.exists():
            print(f"  [{i}/{len(keys)}] sub-{subj} (already done, skip)")
            continue

        print(f"  [{i}/{len(keys)}] sub-{subj}")
        tmp_nii = tmp_dir / f"{dataset_key}_sub-{subj}.nii.gz"
        try:
            print(f"    [download] s3://{BUCKET}/{key}", flush=True)
            download_file(s3, key, tmp_nii)
            print(f"    [downsample] ...", flush=True)
            ds = downsample_4d(tmp_nii)
            out_subj_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ds, out_path)
            print(f"    [saved] {out_path}  shape={tuple(ds.shape)}")
        except Exception as e:
            print(f"    [error] sub-{subj}: {e}")
        finally:
            if tmp_nii.exists():
                tmp_nii.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Where the downsampled .pt files go.")
    ap.add_argument("--tmp-dir", default="/tmp/aomic_raw",
                    help="Temp dir for raw NIfTIs (deleted after each session).")
    ap.add_argument("--datasets", default="piop1,piop2",
                    help="Comma-separated list: piop1, piop2, or both (default).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N subjects per dataset (testing).")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in requested:
        if d not in DATASETS:
            print(f"ERROR: unknown dataset {d!r}. Choices: {list(DATASETS)}",
                  file=sys.stderr)
            sys.exit(2)

    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target shape : {TARGET_SHAPE}")
    print(f"Output       : {output_dir}")
    print(f"Tmp          : {tmp_dir}")
    print(f"Datasets     : {requested}")

    s3 = make_s3_client()

    for d in requested:
        process_dataset(s3, d, DATASETS[d], output_dir, tmp_dir, args.limit)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
