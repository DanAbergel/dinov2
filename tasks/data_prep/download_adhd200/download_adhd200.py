"""Download ADHD-200 preprocessed rs-fMRI (Neuro Bureau / CPAC) from the public
fcp-indi S3 bucket (ANONYMOUS, no login) and downsample to (45,54,45), matching
our other datasets. Also fetches the per-site phenotypic CSVs (ADHD diagnosis,
age, sex, site) merged into one file.

Confirmed S3 layout (2026-07, verified against the bucket):
  imaging: data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/
             {SUBJ}/functional_mni/_scan_rest_1/.../..global0../residual_antswarp.nii.gz
           = 4D BOLD in MNI, no global-signal regression (closest to our minimal
             preprocessing, like ABIDE cpac/nofilt_noglobal).
  labels:  data/Projects/ADHD200/RawDataBIDS/<SITE>_phenotypic.csv

Usage:
    python download_adhd200.py \
        --output-dir /sci/labs/arieljaffe/dan.abergel1/ADHD200_data/downsampled \
        --pheno-out  /sci/labs/arieljaffe/dan.abergel1/ADHD200_data/adhd200_phenotypic.csv \
        --tmp-dir /tmp/adhd_raw
"""

import argparse
import csv
import io
import shutil
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from botocore import UNSIGNED
from botocore.client import Config

BUCKET = "fcp-indi"
IMG_PREFIX = "data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark/"
PHENO_PREFIX = "data/Projects/ADHD200/RawDataBIDS/"
IMG_SUFFIX = "residual_antswarp.nii.gz"
TARGET_SHAPE = (45, 54, 45)


def make_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_img_keys(cli):
    """One 4D BOLD per subject: global0 (no GSR), rest_1."""
    keys, pag = [], cli.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=IMG_PREFIX):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith(IMG_SUFFIX) and "global0" in k and "_scan_rest_1" in k:
                keys.append(k)
    return sorted(keys)


def subj_from_key(k):
    return k.split(IMG_PREFIX)[1].split("/")[0]        # e.g. 0010020_session_1


def downsample_4d(nii_path):
    data = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"expected 4D, got {data.shape}")
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)   # (1,T,X,Y,Z)
    return F.interpolate(vol, size=TARGET_SHAPE, mode="trilinear",
                         align_corners=False).squeeze(0)             # (T,45,54,45)


def download_pheno(cli, out):
    r = cli.list_objects_v2(Bucket=BUCKET, Prefix=PHENO_PREFIX, Delimiter="/")
    header, rows = None, []
    for o in sorted(r.get("Contents", []), key=lambda x: x["Key"]):
        k = o["Key"]
        if not k.endswith("_phenotypic.csv"):
            continue
        site = Path(k).name.replace("_phenotypic.csv", "")
        body = cli.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode("utf-8", "replace")
        rr = list(csv.reader(io.StringIO(body)))
        if not rr:
            continue
        if header is None:
            header = rr[0] + ["SITE"]
        for row in rr[1:]:
            if any(c.strip() for c in row):
                rows.append(row + [site])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"phenotypic: {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir")
    ap.add_argument("--pheno-out")
    ap.add_argument("--tmp-dir", default="/tmp/adhd_raw")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cli = make_s3()
    if args.pheno_out:
        download_pheno(cli, args.pheno_out)
    if not args.output_dir:
        return

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    keys = list_img_keys(cli)
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} ADHD-200 subjects (global0, rest_1)  -> {out}")
    for i, k in enumerate(keys, 1):
        subj = subj_from_key(k)
        op = out / f"{subj}_downsampled.pt"
        if op.exists():
            print(f"  [{i}/{len(keys)}] {subj} (skip)")
            continue
        tn = tmp / f"{subj}.nii.gz"
        try:
            cli.download_file(BUCKET, k, str(tn))
            ds = downsample_4d(tn)
            torch.save(ds, op)
            print(f"  [{i}/{len(keys)}] {subj}  {tuple(ds.shape)}")
        except Exception as e:
            print(f"  [error] {subj}: {e}")
        finally:
            if tn.exists():
                tn.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
