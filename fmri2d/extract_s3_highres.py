"""Stream FULL-RESOLUTION HCP rfMRI volumes from S3, extract 2D PNG slices, discard the volume.

Our local HCP is downsampled 2x/axis (45x54x45); the S3 original is 91x109x91 @ 2mm.
For each subject: `aws s3 cp` the original .nii.gz -> extract slice(s) with extract_slices.py
helpers -> delete the .nii.gz (so disk never holds more than one volume at a time).

Output is an ImageFolder (out/<class>/subject_<id>_rfMRI_REST1_LR[_tNN].png) directly comparable
to brain2d / brain2d_frames but at 2x the spatial detail. Then train/probe with ablation.sh on it.

Sharding: pass --shard i --n-shards N to process subjects[i::N] (drives a SLURM job array).

Needs: awscli + HCP creds, nibabel, pillow, numpy (torch_env). Run via extract_s3_highres.sh.
"""
import argparse
import glob
import os
import subprocess

import numpy as np

from extract_slices import (best_slice_index, frame_volumes, load_4d,
                            load_volume, save_slice, _write_png)


def list_subjects(ds_dir):
    subs = []
    for d in sorted(glob.glob(os.path.join(ds_dir, "subject_*"))):
        if os.path.isdir(d):
            subs.append(os.path.basename(d).split("subject_", 1)[1])
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-dir", required=True, help="dir with subject_<id>/ folders (our local HCP)")
    ap.add_argument("--out", required=True, help="ImageFolder output root")
    ap.add_argument("--tmp", required=True, help="scratch dir for the streamed volume (deleted after each)")
    ap.add_argument("--class-name", default="HCP")
    ap.add_argument("--axis", type=int, default=2, help="0=sag 1=cor 2=axial (default)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-frames", type=int, default=0, help=">0: N slices at N timepoints (same slice idx); 0: temporal mean")
    ap.add_argument("--s3-template",
                    default="s3://hcp-openaccess/HCP_1200/{sid}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz")
    ap.add_argument("--limit", type=int, default=0, help="only the first N subjects (0 = all) — for a quick test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    subs = list_subjects(args.subjects_dir)
    if args.limit:
        subs = subs[: args.limit]
    subs = subs[args.shard :: args.n_shards]
    cls = args.class_name
    os.makedirs(os.path.join(args.out, cls), exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    print(f"shard {args.shard}/{args.n_shards}: {len(subs)} subjects  axis={args.axis} "
          f"size={args.size} n_frames={args.n_frames}", flush=True)

    done = skipped = failed = 0
    for i, sid in enumerate(subs):
        stem = f"subject_{sid}_rfMRI_REST1_LR"
        first_png = os.path.join(args.out, cls, f"{stem}{'_t00' if args.n_frames else ''}.png")
        if os.path.exists(first_png):        # resumable: already extracted
            skipped += 1
            continue
        s3 = args.s3_template.format(sid=sid)
        local = os.path.join(args.tmp, f"subject_{sid}.nii.gz")
        try:
            subprocess.run(["aws", "s3", "cp", s3, local, "--quiet"], check=True)
        except subprocess.CalledProcessError:
            print(f"  SKIP {sid}: s3 cp failed ({s3})", flush=True)
            failed += 1
            continue
        try:
            if args.n_frames > 0:
                arr, t_first = load_4d(local)
                if arr.ndim != 4:
                    save_slice(arr, args.axis, args.out, cls, f"{stem}_t00", args.size)
                else:
                    zidx = best_slice_index(arr.mean(axis=(0 if t_first else -1)), args.axis)
                    for j, vt in enumerate(frame_volumes(arr, t_first, args.n_frames)):
                        _write_png(np.take(vt, zidx, axis=args.axis), args.out, cls, f"{stem}_t{j:02d}", args.size)
            else:
                data = load_volume(local, "mean")
                save_slice(data, args.axis, args.out, cls, stem, args.size)
            done += 1
        except Exception as e:
            print(f"  SKIP {sid}: extract failed ({e})", flush=True)
            failed += 1
        finally:
            if os.path.exists(local):
                os.remove(local)
        if i < 5 or (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(subs)} {sid}  (done={done} skip={skipped} fail={failed})", flush=True)

    print(f"DONE shard {args.shard}: extracted={done} already={skipped} failed={failed} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
