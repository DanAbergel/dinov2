"""Download UCLA CNP (Consortium for Neuropsychiatric Phenomics, ds000030)
preprocessed rest-fMRI from the public OpenNeuro S3 bucket (ANONYMOUS, no login)
and downsample to (45,54,45). Same-dataset comparison to NeuroSTORM (UCLA).

Uses the fMRIPrep-preprocessed derivatives in MNI space (Gorgolewski et al. 2017,
"Preprocessed CNP", ds000030 R1.0.5):
  s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/fmriprep/
      sub-XXXXX/func/sub-XXXXX_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
  labels: .../uncompressed/participants.tsv  (diagnosis: CONTROL/SCHZ/BIPOLAR/ADHD)

265 subjects: 130 control, 50 schizophrenia, 49 bipolar, 43 ADHD.

Usage:
    python tasks/data_prep/download_ucla/download_ucla.py \
        --output-dir /sci/labs/arieljaffe/dan.abergel1/UCLA_data/downsampled \
        --labels-out /sci/labs/arieljaffe/dan.abergel1/UCLA_data/ucla_participants.tsv \
        --tmp-dir /tmp/ucla_raw
"""

import argparse
import shutil
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from botocore import UNSIGNED
from botocore.client import Config

BUCKET = "openneuro"
ROOT = "ds000030/ds000030_R1.0.5/uncompressed"
IMG_PREFIX = f"{ROOT}/derivatives/fmriprep/"
IMG_SUFFIX = "task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
PARTICIPANTS = f"{ROOT}/participants.tsv"
TARGET_SHAPE = (45, 54, 45)


def make_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_img_keys(cli):
    keys, pag = [], cli.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=IMG_PREFIX):
        for o in page.get("Contents", []):
            if o["Key"].endswith(IMG_SUFFIX):
                keys.append(o["Key"])
    return sorted(keys)


def subj_from_key(k):
    # .../fmriprep/sub-10159/func/sub-10159_task-rest_..._preproc.nii.gz -> sub-10159
    return k.split(IMG_PREFIX)[1].split("/")[0]


def downsample_4d(nii_path):
    data = nib.load(str(nii_path)).get_fdata(dtype=np.float32)      # (X,Y,Z,T)
    if data.ndim != 4:
        raise ValueError(f"expected 4D, got {data.shape}")
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)   # (1,T,X,Y,Z)
    return F.interpolate(vol, size=TARGET_SHAPE, mode="trilinear",
                         align_corners=False).squeeze(0)             # (T,45,54,45)


def download_labels(cli, out):
    body = cli.get_object(Bucket=BUCKET, Key=PARTICIPANTS)["Body"].read()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(body)
    n = body.decode("utf-8", "replace").count("\n")
    print(f"participants.tsv: ~{n} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--labels-out", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/ucla_raw")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    cli = make_s3()
    download_labels(cli, args.labels_out)

    keys = list_img_keys(cli)
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} UCLA subjects (rest, MNI preproc) -> {out}")
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
            print(f"  [{i}/{len(keys)}] {subj}  {tuple(ds.shape)}", flush=True)
        except Exception as e:
            print(f"  [error] {subj}: {e}", flush=True)
        finally:
            if tn.exists():
                tn.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
