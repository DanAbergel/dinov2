"""Download HCP-YA rfMRI from AWS S3 and downsample to (45, 54, 45) on the fly.

Streams one session at a time:
  1. Download raw 4D NIfTI (~1.5 GB) from s3://hcp-openaccess
  2. Resample to (45, 54, 45) with trilinear interpolation
  3. Save downsampled tensor as .pt
  4. Delete raw NIfTI to free disk

For each subject, downloads only the REST1_LR session by default (1 of 4
possible runs). Pass --all-sessions to grab all 4.

Usage on Moriah:
    python scripts/download_and_downsample_hcp.py \
        --subjects-csv data/HCP_YA_subjects.csv \
        --output-dir /sci/labs/arieljaffe/dan.abergel1/HCP_data/downsampled_v2 \
        --tmp-dir /tmp/hcp_raw \
        --aws-profile hcp
"""

import argparse
import os
import shutil
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from botocore.config import Config

BUCKET = "hcp-openaccess"
SESSION_TEMPLATE = "HCP_1200/{subject}/MNINonLinear/Results/{run}/{run}.nii.gz"
DEFAULT_RUNS = ["rfMRI_REST1_LR"]   # 1 run per subject by default
ALL_RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]

TARGET_SHAPE = (45, 54, 45)         # (X, Y, Z) — matches our pipeline


def make_s3_client(profile_name: str):
    """boto3 client with requester-pays enabled for hcp-openaccess."""
    session = boto3.Session(profile_name=profile_name)
    return session.client(
        "s3",
        config=Config(
            request_checksum_calculation="when_required",
            signature_version="s3v4",
        ),
    )


def download_run(s3, subject: str, run: str, dest: Path):
    """Download one HCP run's 4D NIfTI from S3 (requester-pays)."""
    key = SESSION_TEMPLATE.format(subject=subject, run=run)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(
        Bucket=BUCKET,
        Key=key,
        Filename=str(dest),
        ExtraArgs={"RequestPayer": "requester"},
    )


def downsample_4d(nii_path: Path) -> torch.Tensor:
    """Load 4D NIfTI, downsample spatial to TARGET_SHAPE via trilinear.

    Returns float32 tensor of shape (T, X', Y', Z') = (T, 45, 54, 45).
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)          # (X, Y, Z, T) in MNI 2mm
    # Move T to front and add batch + channel dims for F.interpolate:
    # (X, Y, Z, T) -> (T, X, Y, Z) -> (1, T, X, Y, Z) -> (1, T, 1, X, Y, Z)? No.
    # F.interpolate expects (N, C, D, H, W) for 3D = (1, T, X, Y, Z) where T = channels.
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)   # (1, T, X, Y, Z)
    vol_ds = F.interpolate(
        vol,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return vol_ds.squeeze(0)                                          # (T, 45, 54, 45)


def process_subject(s3, subject: str, runs, output_dir: Path, tmp_dir: Path):
    """Download -> downsample -> save -> delete raw, per run."""
    out_subject_dir = output_dir / f"subject_{subject}"
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        out_path = out_subject_dir / f"{run}_downsampled.pt"
        if out_path.exists():
            print(f"  [skip] {subject}/{run} (already done)")
            continue

        tmp_nii = tmp_dir / f"{subject}_{run}.nii.gz"
        print(f"  [download] {subject}/{run} ...", flush=True)
        try:
            download_run(s3, subject, run, tmp_nii)
        except Exception as e:
            print(f"  [error] {subject}/{run} download failed: {e}")
            continue

        print(f"  [downsample] {subject}/{run} ...", flush=True)
        try:
            ds = downsample_4d(tmp_nii)
            torch.save(ds, out_path)
            print(f"  [saved] {out_path}  shape={tuple(ds.shape)} dtype={ds.dtype}")
        except Exception as e:
            print(f"  [error] {subject}/{run} downsample failed: {e}")
        finally:
            # Free disk regardless of outcome
            if tmp_nii.exists():
                tmp_nii.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-csv", required=True,
                    help="CSV with a 'Subject' column listing HCP subject IDs.")
    ap.add_argument("--output-dir", required=True,
                    help="Where the downsampled .pt files go.")
    ap.add_argument("--tmp-dir", default="/tmp/hcp_raw",
                    help="Temp dir for raw NIfTIs (deleted after each session).")
    ap.add_argument("--aws-profile", default="hcp",
                    help="AWS credentials profile (~/.aws/credentials).")
    ap.add_argument("--all-sessions", action="store_true",
                    help="Download all 4 runs per subject (default: just REST1_LR).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N subjects (for testing).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(args.subjects_csv)["Subject"].astype(str).tolist()
    if args.limit:
        subjects = subjects[:args.limit]
    runs = ALL_RUNS if args.all_sessions else DEFAULT_RUNS
    print(f"Subjects: {len(subjects)} | runs/subject: {runs} | target: {TARGET_SHAPE}")

    s3 = make_s3_client(args.aws_profile)

    for i, subj in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] subject {subj}")
        process_subject(s3, subj, runs, output_dir, tmp_dir)

    # Clean up tmp_dir at the end
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
