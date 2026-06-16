"""Downsample OASIS-3 raw NIfTIs to (T, 45, 54, 45) .pt.

Input directory layout (produced by the NrgXnat download_oasis_scans.sh script):
    raw_nifti/
      <experiment_id>/
        <scan_type_dir>/                # e.g. func1, func2, func3, anat1, ...
          <files>.nii.gz                # the BIDS-style filenames
        ...

For each experiment_id we keep the EARLIEST run of resting-state BOLD
(typically run-01), downsample (X, Y, Z, T) -> (T, 45, 54, 45) via
trilinear interpolation, and save as a single .pt per subject.

Output:
    OASIS3_data/downsampled/
      OAS30001/rest_d0129_downsampled.pt
      OAS30002/rest_d0241_downsampled.pt
      ...
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

TARGET_SHAPE = (45, 54, 45)

# OASIS-3 BIDS pattern for the rest BOLD files inside each func dir:
#   sub-OAS30001_ses-d0129_task-rest_run-01_bold.nii.gz
REST_PATTERN = re.compile(
    r"sub-(OAS\d+)_ses-(d\d+)_task-rest(?:_run-(\d+))?_bold\.nii\.gz$"
)


def downsample_4d(nii_path: Path) -> torch.Tensor:
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    # DEBUG: print native shape to diagnose unexpected T (e.g. T=5).
    print(f"    native nibabel shape: {data.shape}  (file: {nii_path.name})")
    if data.ndim != 4:
        raise ValueError(f"expected 4D (X,Y,Z,T), got {data.ndim}D {data.shape}")
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, X, Y, Z)
    vol_ds = F.interpolate(
        vol, size=TARGET_SHAPE, mode="trilinear", align_corners=False,
    )
    return vol_ds.squeeze(0)


def find_first_rest(experiment_dir: Path):
    """Return the rest-BOLD run with the MOST timepoints.

    OASIS-3 sessions often have 3 rest runs where run-01 is a short
    calibration (~5 frames) and the real rs-fMRI is a longer run. So we
    read each candidate's nibabel header (cheap — no data load) and pick
    the one with the largest T.
    """
    candidates = []
    for func_dir in experiment_dir.glob("func*"):
        for f in func_dir.glob("*.nii.gz"):
            m = REST_PATTERN.match(f.name)
            if m:
                candidates.append((f, m.group(1), m.group(2)))
    if not candidates:
        return None

    best = None  # (T, path, subject, day)
    for path, subject, day in candidates:
        try:
            shape = nib.load(str(path)).shape          # header only, no data
        except Exception:
            continue
        T = shape[3] if len(shape) == 4 else 0
        if best is None or T > best[0]:
            best = (T, path, subject, day)
    if best is None:
        return None
    T, path, subject, day = best
    if T < 20:
        print(f"    WARNING: longest rest run for {subject}/{day} has only "
              f"T={T} frames")
    return path, subject, day


def read_tr_from_json(nii_path: Path):
    """Read RepetitionTime (seconds) from the BIDS JSON sidecar next to the
    NIfTI. OASIS-3 has no fixed TR, so this per-scan value is essential and
    MUST be captured before the raw is deleted. Returns None if unavailable.
    """
    json_path = Path(str(nii_path).replace(".nii.gz", ".json"))
    if not json_path.exists():
        return None
    try:
        with open(json_path) as f:
            meta = json.load(f)
        return meta.get("RepetitionTime")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Dir with per-experiment subfolders from NrgXnat")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--delete-raw", action="store_true",
                    help="Delete each experiment's raw dir after its .pt is "
                         "saved (frees disk; the TR is captured to the manifest "
                         "first).")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.exists():
        print(f"ERROR: input dir {input_dir} missing", file=sys.stderr)
        sys.exit(2)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest of per-scan TR (OASIS-3 has no fixed TR). Appended as we go so
    # the TR survives raw deletion. Format: subject,day,tr,T
    manifest_path = output_dir / "oasis3_tr_manifest.csv"
    if not manifest_path.exists():
        manifest_path.write_text("subject,day,tr,T\n")

    experiments = sorted(p for p in input_dir.iterdir() if p.is_dir())
    print(f"Found {len(experiments)} experiment dirs to process.")
    print(f"delete-raw: {args.delete_raw}")
    n_ok = n_skip = n_err = 0

    for i, exp_dir in enumerate(experiments, 1):
        found = find_first_rest(exp_dir)
        if found is None:
            print(f"[{i}/{len(experiments)}] {exp_dir.name}: no rest BOLD")
            n_err += 1
            continue
        nii_path, subject, day = found
        out_subject_dir = output_dir / subject
        out_path = out_subject_dir / f"rest_{day}_downsampled.pt"
        if out_path.exists():
            print(f"[{i}/{len(experiments)}] {subject}/{day}: skip (already done)")
            n_skip += 1
            if args.delete_raw:                      # already done -> free its raw
                shutil.rmtree(exp_dir, ignore_errors=True)
            continue
        try:
            tr = read_tr_from_json(nii_path)          # capture TR BEFORE delete
            ds = downsample_4d(nii_path)
            out_subject_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ds, out_path)
            with open(manifest_path, "a") as f:
                f.write(f"{subject},{day},{tr},{ds.shape[0]}\n")
            print(f"[{i}/{len(experiments)}] {subject}/{day}: "
                  f"saved shape={tuple(ds.shape)} tr={tr}")
            n_ok += 1
            if args.delete_raw:
                shutil.rmtree(exp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[{i}/{len(experiments)}] {subject}/{day}: error {e!r}")
            n_err += 1

    print(f"\nSummary: ok={n_ok} skipped={n_skip} errors={n_err}")
    print(f"TR manifest: {manifest_path}")


if __name__ == "__main__":
    main()
