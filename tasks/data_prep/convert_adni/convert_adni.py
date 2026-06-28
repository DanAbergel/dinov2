"""Convert Sagi's stacked ADNI tensor into per-scan .pt files matching the
rest of the corpus (HCP / ABIDE / OASIS-3 / AOMIC).

Sagi stores ALL ADNI scans in one monolithic tensor:
    all_4d_downsampled.pt  -> shape (N, X, Y, Z, T)   (T is the LAST axis)
plus two JSON sidecars linking the integer scan index to clinical labels:
    index_to_name.json      :  "0"   -> {image_id: "I123", subject_id: "002_S_0413"}
    imageID_to_labels.json  :  "I123"-> {AGE, MMSE, CDR, degradation_binary_1/2/3, ...}

The other four datasets are stored as ONE .pt per scan, shape (T, 45, 54, 45),
grouped per subject. This script makes ADNI uniform with them so a single
MixedFMRIDataset loader can glob `*.pt` across all five datasets.

Output:
    ADNI_data/downsampled/
      <subject_id>/<image_id>.pt        # tensor (T, 45, 54, 45), float32
      ...
    ADNI_data/downsampled/adni_manifest.csv
      image_id,subject_id,T,tr,<label columns...>

Notes:
  * The T axis is auto-detected (the dim whose size is the outlier vs the
    spatial triplet) and permuted to the FRONT, exactly like adni_probe.py.
  * Spatial dims are trilinearly resampled to (45, 54, 45) only if they
    differ — Sagi's main file is expected to already be 45x54x45.
  * TR: ADNI single-band rs-fMRI is 3.0 s (multi-band 0.607 s). Sagi's tensor
    does not carry per-scan TR, so we write a single assumed value (--tr,
    default 3.0). Adjust later if a per-scan TR table becomes available.

Usage (one-shot, CPU is fine):
    python convert_adni.py \
        --source /sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data \
        --output-dir /sci/labs/arieljaffe/dan.abergel1/ADNI_data/downsampled
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

TARGET_SHAPE = (45, 54, 45)
SOURCE_TENSOR = "all_4d_downsampled.pt"
INDEX_JSON = "index_to_name.json"
LABELS_JSON = "imageID_to_labels.json"


def detect_T_position(shape) -> bool:
    """Return True if T is the LAST axis (N, X, Y, Z, T), False if FIRST.

    The spatial triplet (X, Y, Z) has similar magnitudes; T is the outlier.
    We compare the spread (max/min) of the last-three vs first-three of the
    non-batch dims. Works from the SHAPE alone — no data is moved (critical
    so we never materialize the full multi-GB tensor in RAM).
    """
    if len(shape) != 5:
        raise ValueError(f"expected 5D (N,a,b,c,d), got {len(shape)}D {tuple(shape)}")
    rest = list(shape[1:])  # 4 dims (a,b,c,d)
    spread_if_T_last = max(rest[:-1]) / min(rest[:-1])   # treat last as T
    spread_if_T_first = max(rest[1:]) / min(rest[1:])    # treat first as T
    t_is_last = spread_if_T_last < spread_if_T_first
    print(f"  T detected {'LAST' if t_is_last else 'FIRST'} "
          f"(per-scan dims {tuple(rest)})")
    return t_is_last


def resample_spatial(vol_4d: torch.Tensor) -> torch.Tensor:
    """(T, X, Y, Z) -> (T, 45, 54, 45) via trilinear, only if needed."""
    if tuple(vol_4d.shape[1:]) == TARGET_SHAPE:
        return vol_4d
    out = F.interpolate(
        vol_4d.unsqueeze(0), size=TARGET_SHAPE, mode="trilinear", align_corners=False,
    ).squeeze(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Sagi's shared_fmri_data dir (holds the tensor + JSONs)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tr", type=float, default=3.0,
                    help="Assumed TR in seconds (ADNI single-band = 3.0)")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # mmap=True keeps the tensor on disk and reads only the slice we index
    # (data[i]) into RAM. Essential: the stacked tensor can be tens of GB,
    # which OOM-kills a naive full load.
    print(f"Loading {src/SOURCE_TENSOR} (mmap, lazy) ...")
    data = torch.load(src / SOURCE_TENSOR, map_location="cpu",
                      weights_only=True, mmap=True)
    print(f"  raw shape={tuple(data.shape)} dtype={data.dtype}")
    t_is_last = detect_T_position(data.shape)

    with open(src / INDEX_JSON) as f:
        index_to_name = json.load(f)
    with open(src / LABELS_JSON) as f:
        image_labels = json.load(f)

    N = data.shape[0]
    if len(index_to_name) != N:
        print(f"  WARNING: tensor has {N} scans but index_to_name has "
              f"{len(index_to_name)} entries")

    # T is uniform across scans (a stacked tensor forces the same length for
    # all), so read it once from the shape — no need to touch each scan.
    T_global = data.shape[-1] if t_is_last else data.shape[1]
    print(f"  uniform T = {T_global}")

    # ---- STEP 1: write the FULL manifest up front, from the JSONs alone. ----
    # Independent of the (slow, killable) volume loop, so the labels are always
    # complete on disk even if volume conversion is interrupted. Flushed+closed
    # immediately (the old version buffered rows until the end -> empty file on
    # a kill).
    label_keys = set()
    for v in image_labels.values():
        label_keys.update(v.keys())
    label_keys = sorted(label_keys)

    manifest_path = out / "adni_manifest.csv"
    with open(manifest_path, "w", newline="") as f_manifest:
        writer = csv.writer(f_manifest)
        writer.writerow(["image_id", "subject_id", "T", "tr"] + label_keys)
        for idx in sorted(index_to_name.keys(), key=int):
            entry = index_to_name[idx]
            image_id = entry["image_id"]
            labels = image_labels.get(image_id, {})
            writer.writerow([image_id, entry["subject_id"], T_global, args.tr]
                            + [labels.get(k, "") for k in label_keys])
        f_manifest.flush()
    print(f"  manifest written ({len(index_to_name)} rows): {manifest_path}")

    # ---- STEP 2: convert volumes (skippable / resumable). ----
    n_ok = n_skip = 0
    for idx in sorted(index_to_name.keys(), key=int):
        i = int(idx)
        entry = index_to_name[idx]
        image_id = entry["image_id"]
        subject_id = entry["subject_id"]

        subj_dir = out / subject_id
        out_path = subj_dir / f"{image_id}.pt"
        if out_path.exists():
            n_skip += 1
        else:
            # Read ONLY scan i from the mmap'd file, then move T to front
            # per-scan (a single scan is small, so permute/contiguous is cheap).
            vol = data[i].float()                           # (X,Y,Z,T) or (T,X,Y,Z)
            if t_is_last:
                vol = vol.permute(3, 0, 1, 2).contiguous()  # -> (T, X, Y, Z)
            vol = resample_spatial(vol)                     # (T, 45, 54, 45)
            subj_dir.mkdir(parents=True, exist_ok=True)
            torch.save(vol.contiguous(), out_path)
            n_ok += 1

        if (n_ok + n_skip) % 100 == 0:
            print(f"  {n_ok+n_skip}/{N}  (saved={n_ok} skip={n_skip})")

    print(f"\nDone: saved={n_ok} skipped={n_skip} of {N}")
    print(f"Per-scan .pt under: {out}")
    print(f"Manifest: {manifest_path}")
    # quick subject count
    n_subj = len({index_to_name[k]['subject_id'] for k in index_to_name})
    print(f"Unique subjects: {n_subj}")


if __name__ == "__main__":
    main()
