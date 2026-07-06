"""Download HCP-YA TASK-fMRI from AWS S3 and downsample to (45,54,45) on the fly.

This is the NeuroSTORM 'task-state classification' benchmark: one sample = one
task run, label = which of the 7 HCP tasks it is (7-class classification).

Same S3 layout / requester-pays / downsampling as download_hcp.py (rest), only
the run names change:  tfMRI_{TASK}_{LR|RL}  instead of rfMRI_REST*.

  s3://hcp-openaccess/HCP_1200/{subject}/MNINonLinear/Results/tfMRI_{TASK}_{PE}/tfMRI_{TASK}_{PE}.nii.gz

Streams one run at a time: download raw 4D NIfTI -> trilinear resample to
(45,54,45) -> save .pt -> delete raw. By default one phase-encoding (LR) per
task -> 7 runs/subject; pass --all-pe for LR+RL (14/subject).

A manifest CSV (subject, task, pe, path) is written so the probe can read the
task label without re-listing S3.

Usage on Moriah (needs FRESH ConnectomeDB S3 keys — the old ones expire):
    python tasks/data_prep/download_hcp_task/download_hcp_task.py \
        --output-dir /sci/labs/arieljaffe/dan.abergel1/HCP_task_data/downsampled \
        --manifest   /sci/labs/arieljaffe/dan.abergel1/HCP_task_data/hcp_task_labels.csv \
        --tmp-dir /tmp/hcp_task_raw --aws-profile default
    # confirm the S3 paths first on a couple of subjects:
    python .../download_hcp_task.py --output-dir ... --manifest ... --limit 2
"""

import argparse
import csv
import shutil
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from botocore.config import Config

BUCKET = "hcp-openaccess"
RUN_TEMPLATE = "HCP_1200/{subject}/MNINonLinear/Results/{run}/{run}.nii.gz"
TASKS = ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
TARGET_SHAPE = (45, 54, 45)


def make_s3_client(profile_name):
    session = boto3.Session(profile_name=profile_name)
    return session.client(
        "s3",
        config=Config(request_checksum_calculation="when_required",
                      signature_version="s3v4"),
    )


def list_subjects(s3):
    pag = s3.get_paginator("list_objects_v2")
    subjects = set()
    for page in pag.paginate(Bucket=BUCKET, Prefix="HCP_1200/", Delimiter="/",
                             RequestPayer="requester"):
        for c in page.get("CommonPrefixes", []):
            subj = c["Prefix"].rstrip("/").split("/")[-1]
            if subj.isdigit():
                subjects.add(subj)
    return sorted(subjects)


def download_run(s3, subject, run, dest):
    key = RUN_TEMPLATE.format(subject=subject, run=run)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(Bucket=BUCKET, Key=key, Filename=str(dest),
                     ExtraArgs={"RequestPayer": "requester"})


def downsample_4d(nii_path):
    data = nib.load(str(nii_path)).get_fdata(dtype=np.float32)   # (X,Y,Z,T)
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)  # (1,T,X,Y,Z)
    return F.interpolate(vol, size=TARGET_SHAPE, mode="trilinear",
                         align_corners=False).squeeze(0)           # (T,45,54,45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest", required=True,
                    help="CSV written with subject,task,pe,path for the probe.")
    ap.add_argument("--tmp-dir", default="/tmp/hcp_task_raw")
    ap.add_argument("--aws-profile", default="default")
    ap.add_argument("--all-pe", action="store_true",
                    help="Both phase-encodings LR+RL (default: LR only).")
    ap.add_argument("--tasks", nargs="+", default=TASKS,
                    help=f"Subset of tasks (default: all 7 {TASKS}).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    s3 = make_s3_client(args.aws_profile)
    print(f"Listing HCP_1200/ on s3://{BUCKET} ...")
    subjects = list_subjects(s3)
    print(f"Discovered {len(subjects)} subjects.")
    if args.limit:
        subjects = subjects[:args.limit]
    pes = ["LR", "RL"] if args.all_pe else ["LR"]
    print(f"{len(subjects)} subjects | tasks: {args.tasks} | PE: {pes} | target {TARGET_SHAPE}")

    manifest = []
    for i, subj in enumerate(subjects, 1):
        sdir = out / f"subject_{subj}"; sdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(subjects)}] subject {subj}")
        for task in args.tasks:
            for pe in pes:
                run = f"tfMRI_{task}_{pe}"
                op = sdir / f"{run}_downsampled.pt"
                if op.exists():
                    print(f"  [skip] {run}")
                    manifest.append([subj, task, pe, str(op)])
                    continue
                tn = tmp / f"{subj}_{run}.nii.gz"
                try:
                    download_run(s3, subj, run, tn)
                    ds = downsample_4d(tn)
                    torch.save(ds, op)
                    manifest.append([subj, task, pe, str(op)])
                    print(f"  [saved] {run}  {tuple(ds.shape)}")
                except Exception as e:
                    print(f"  [error] {run}: {e}")
                finally:
                    if tn.exists():
                        tn.unlink()
        # rewrite manifest each subject so a crash still leaves a usable CSV
        with open(args.manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "task", "pe", "path"])
            w.writerows(manifest)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nDone. manifest: {len(manifest)} runs -> {args.manifest}")


if __name__ == "__main__":
    main()
