"""Download COBRE schizophrenia rs-fMRI (preprocessed with NIAK) from figshare
and downsample to (45,54,45). Fully autonomous — public figshare, no login/S3.

NeuroSTORM disease benchmark: schizophrenia vs control (146 subjects, ~72 SZ /
74 HC). Source = figshare article 1160600 "COBRE preprocessed with NIAK 0.12.4":
  - 146 x fmri_{SUBJ}_session1_run1.nii.gz   (4D BOLD, MNI)
  - cobre_model_group.csv : subject, sz (1=schizophrenia, 0=control), age, sex, FD

Streams one subject at a time: download nii.gz -> trilinear resample to
(45,54,45) -> save .pt -> delete raw. Writes cobre_labels.csv for the probe.

Usage:
    python tasks/data_prep/download_cobre/download_cobre.py \
        --output-dir /sci/labs/arieljaffe/dan.abergel1/COBRE_data/downsampled \
        --labels-out /sci/labs/arieljaffe/dan.abergel1/COBRE_data/cobre_labels.csv \
        --tmp-dir /tmp/cobre_raw
"""

import argparse
import csv
import io
import json
import shutil
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ARTICLE = 1160600
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
LABEL_FILE = "cobre_model_group.csv"
TARGET_SHAPE = (45, 54, 45)


def figshare_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.load(r)["files"]


def fetch(url, dest, retries=3):
    for k in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            if k == retries - 1:
                raise
            print(f"    retry {k+1}/{retries} ({e})", flush=True)


def subj_from_name(name):
    # fmri_szxxx0040044_session1_run1.nii.gz -> szxxx0040044
    return name[len("fmri_"):].split("_session")[0]


def downsample_4d(nii_path):
    data = nib.load(str(nii_path)).get_fdata(dtype=np.float32)      # (X,Y,Z,T)
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)   # (1,T,X,Y,Z)
    return F.interpolate(vol, size=TARGET_SHAPE, mode="trilinear",
                         align_corners=False).squeeze(0)             # (T,45,54,45)


def write_labels(files, out):
    lab = next((f for f in files if f["name"] == LABEL_FILE), None)
    if lab is None:
        print("  (no cobre_model_group.csv found)")
        return
    with urllib.request.urlopen(lab["download_url"], timeout=60) as r:
        txt = r.read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(txt)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "sz", "age", "sex", "FD"])
        for r in rows[1:]:
            if not r or not r[0].strip():
                continue
            v = [c.strip().strip('"') for c in r]
            w.writerow([v[0], v[1], v[2], v[3], v[4] if len(v) > 4 else ""])
    print(f"labels: {len(rows)-1} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--labels-out", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/cobre_raw")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)

    print(f"Fetching figshare article {ARTICLE} file list ...", flush=True)
    files = figshare_files()
    write_labels(files, args.labels_out)

    niis = sorted([f for f in files if f["name"].endswith(".nii.gz")],
                  key=lambda f: f["name"])
    if args.limit:
        niis = niis[:args.limit]
    print(f"{len(niis)} COBRE subjects -> {out}", flush=True)
    for i, f in enumerate(niis, 1):
        subj = subj_from_name(f["name"])
        op = out / f"{subj}_downsampled.pt"
        if op.exists():
            print(f"  [{i}/{len(niis)}] {subj} (skip)")
            continue
        tn = tmp / f["name"]
        try:
            fetch(f["download_url"], str(tn))
            ds = downsample_4d(tn)
            torch.save(ds, op)
            print(f"  [{i}/{len(niis)}] {subj}  {tuple(ds.shape)}", flush=True)
        except Exception as e:
            print(f"  [error] {subj}: {e}", flush=True)
        finally:
            if tn.exists():
                tn.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
