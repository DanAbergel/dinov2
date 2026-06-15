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
import re
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
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)
    vol_ds = F.interpolate(
        vol, size=TARGET_SHAPE, mode="trilinear", align_corners=False,
    )
    return vol_ds.squeeze(0)


def find_first_rest(experiment_dir: Path):
    """Return the path to the earliest run of task-rest BOLD."""
    candidates = []
    for func_dir in experiment_dir.glob("func*"):
        for f in func_dir.glob("*.nii.gz"):
            m = REST_PATTERN.match(f.name)
            if m:
                run = int(m.group(3)) if m.group(3) else 1
                candidates.append((run, f, m.group(1), m.group(2)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, path, subject, day = candidates[0]
    return path, subject, day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Dir with per-experiment subfolders from NrgXnat")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.exists():
        print(f"ERROR: input dir {input_dir} missing", file=sys.stderr)
        sys.exit(2)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = sorted(p for p in input_dir.iterdir() if p.is_dir())
    print(f"Found {len(experiments)} experiment dirs to process.")
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
            continue
        try:
            ds = downsample_4d(nii_path)
            out_subject_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ds, out_path)
            print(f"[{i}/{len(experiments)}] {subject}/{day}: "
                  f"saved shape={tuple(ds.shape)}")
            n_ok += 1
        except Exception as e:
            print(f"[{i}/{len(experiments)}] {subject}/{day}: error {e!r}")
            n_err += 1

    print(f"\nSummary: ok={n_ok} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
