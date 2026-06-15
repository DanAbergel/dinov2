"""Extract Schaefer-200 ROI time-series from downsampled HCP-YA volumes.

For each subject in --input-dir (containing subject_*/rfMRI_*.pt files),
fetches the Schaefer-2018 atlas (n_rois=200), resamples it to match the
volume shape via nearest-neighbor interpolation (preserves integer ROI
labels), then computes the mean BOLD signal per ROI per timepoint.

Output: one .pt file per (subject, run) with a tensor of shape (T, 200).

Usage:
    # Local on Moriah gateway:
    python tasks/extract_schaefer200/extract_schaefer200.py \\
        --input-dir /sci/labs/arieljaffe/dan.abergel1/HCP_data/downsampled \\
        --output-dir ./schaefer200_hcp

    # Via sbatch (output dir defaults to $SLURM_SUBMIT_DIR/schaefer200_output):
    sbatch tasks/extract_schaefer200/extract_schaefer200.sh
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Volume shape used by our pipeline (HCP-YA downsampled, ADNI, etc.).
TARGET_SHAPE = (45, 54, 45)
N_ROIS = 200


def get_resampled_atlas(verbose: bool = True) -> torch.Tensor:
    """Fetch Schaefer 200 atlas and resample to TARGET_SHAPE via nearest-neighbor.

    nilearn returns a NIfTI in MNI152 2mm space, shape (91, 109, 91). We use
    nearest-neighbor so the integer ROI labels are preserved (no fractional
    values), and downsample to (45, 54, 45) matching our voxel grid.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    if verbose:
        print("Fetching Schaefer-2018 atlas (n_rois=200, yeo_networks=7, 2mm)...")
    atlas = fetch_atlas_schaefer_2018(
        n_rois=N_ROIS,
        yeo_networks=7,
        resolution_mm=2,
    )
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata().astype(np.int64)
    if verbose:
        print(f"  Native atlas shape   : {atlas_data.shape}")

    # Resample via nearest-neighbor (preserves integer labels).
    atlas_t = torch.from_numpy(atlas_data).float().unsqueeze(0).unsqueeze(0)
    atlas_resized = F.interpolate(
        atlas_t,
        size=TARGET_SHAPE,
        mode="nearest",
    )
    atlas_final = atlas_resized.squeeze(0).squeeze(0).long()

    unique = torch.unique(atlas_final)
    if verbose:
        print(f"  Resampled atlas shape: {tuple(atlas_final.shape)}")
        print(f"  Unique labels        : {len(unique)} "
              f"(min={unique.min().item()}, max={unique.max().item()})")
        missing = set(range(1, N_ROIS + 1)) - set(unique.tolist())
        if missing:
            print(f"  WARNING: {len(missing)} ROI label(s) lost in downsampling: "
                  f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    return atlas_final


def extract_roi_timeseries(volume: torch.Tensor, atlas: torch.Tensor) -> torch.Tensor:
    """Compute mean BOLD signal per ROI per timepoint.

    Args:
        volume: (T, X, Y, Z) float32 -- one fMRI scan
        atlas:  (X, Y, Z)    int64    -- 0 = background, 1..N_ROIS = ROI labels

    Returns:
        (T, N_ROIS) float32 tensor. Empty ROIs (no voxels of that label after
        downsampling) get zeros.
    """
    T = volume.shape[0]
    volume_flat = volume.reshape(T, -1).float()      # (T, X*Y*Z)
    atlas_flat = atlas.reshape(-1)                    # (X*Y*Z,)

    timeseries = torch.zeros(T, N_ROIS, dtype=torch.float32)
    for roi_id in range(1, N_ROIS + 1):
        mask = (atlas_flat == roi_id)
        if mask.sum() == 0:
            continue
        timeseries[:, roi_id - 1] = volume_flat[:, mask].mean(dim=1)
    return timeseries


def process_subject(subject_dir: Path, output_dir: Path, atlas: torch.Tensor):
    """Extract ROI time-series for every .pt session of one subject."""
    subj_id = subject_dir.name.replace("subject_", "")
    pt_files = sorted(subject_dir.glob("*.pt"))

    if not pt_files:
        print(f"  [skip] subject_{subj_id} -- no .pt files found")
        return

    for pt_file in pt_files:
        # Strip the '_downsampled' suffix from the input name, e.g.
        # 'rfMRI_REST1_LR_downsampled.pt' -> 'rfMRI_REST1_LR'.
        run_name = pt_file.stem.replace("_downsampled", "")
        out_path = output_dir / f"subject_{subj_id}_{run_name}_schaefer200.pt"
        if out_path.exists():
            print(f"  [skip] subject_{subj_id}/{run_name} (already done)")
            continue

        print(f"  [process] subject_{subj_id}/{run_name} ...", flush=True)
        try:
            volume = torch.load(pt_file, map_location="cpu")
            ts = extract_roi_timeseries(volume, atlas)
            torch.save(ts, out_path)
            print(f"  [saved]   {out_path.name}  shape={tuple(ts.shape)} "
                  f"dtype={ts.dtype}")
        except Exception as e:
            print(f"  [error]   subject_{subj_id}/{run_name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Dir containing subject_<id>/rfMRI_*.pt files")
    ap.add_argument("--output-dir", default="./schaefer200_output",
                    help="Where to save per-subject ROI time-series "
                         "(default: ./schaefer200_output in cwd).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N subjects (testing)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")

    atlas = get_resampled_atlas()

    subject_dirs = sorted(input_dir.glob("subject_*"))
    if args.limit:
        subject_dirs = subject_dirs[:args.limit]
    print(f"Subjects to process: {len(subject_dirs)}")

    for i, subj_dir in enumerate(subject_dirs, 1):
        print(f"\n[{i}/{len(subject_dirs)}] {subj_dir.name}")
        process_subject(subj_dir, output_dir, atlas)

    print("\nDone.")


if __name__ == "__main__":
    main()
