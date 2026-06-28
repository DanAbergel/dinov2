"""OASIS-3 streaming downloader: per-subject download -> downsample -> delete raw.

Designed for parallel sharded execution. Each shard processes a disjoint
subset of subjects (subjects[shard_id::n_shards]). For each subject:
  1. find the earliest MR session that contains a resting-state BOLD scan
  2. download that scan's NIfTI (the longest rest run) to a temp dir
  3. read TR from the BIDS JSON sidecar (OASIS-3 has no fixed TR)
  4. downsample (X,Y,Z,T) -> (T,45,54,45) trilinear, save .pt
  5. delete the temp raw immediately  -> no raw accumulation, low disk

All HTTP via `curl` subprocess (NITRC-IR TLS fingerprinting blocks Python's
requests/urllib). Password is URL-encoded (a '#' must become %23) and read
from env XNAT_PASSWORD or ~/.xnat_password.

Usage (one shard):
    XNAT_USERNAME=danab95 python oasis3_streaming.py \
        --output-dir /.../OASIS3_data/downsampled \
        --tmp-dir /.../tmp/oasis3_shardN \
        --shard-id 0 --n-shards 10
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

XNAT_HOST = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"
TARGET_SHAPE = (45, 54, 45)
RSFMRI_RE = re.compile(r"bold", re.IGNORECASE)


# ---------- password ----------

def get_password_encoded() -> str:
    pw = os.environ.get("XNAT_PASSWORD")
    if not pw:
        pw_file = Path("~/.xnat_password").expanduser()
        if not pw_file.exists():
            print("ERROR: no XNAT password (set XNAT_PASSWORD or ~/.xnat_password)",
                  file=sys.stderr)
            sys.exit(2)
        pw = pw_file.read_text().strip()
    # URL-encode (NITRC expects '#' -> %23 etc. in the Basic Auth header)
    return urllib.parse.quote(pw, safe="")


# ---------- curl helpers ----------

def curl_auth(username: str, password_enc: str, cookie_jar: Path) -> bool:
    cmd = ["curl", "-f", "-k", "-s", "-u", f"{username}:{password_enc}",
           "--cookie-jar", str(cookie_jar), f"{XNAT_HOST}/data/JSESSION"]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def curl_json(url: str, cookie_jar: Path):
    r = subprocess.run(
        ["curl", "-f", "-k", "-s", "--cookie", str(cookie_jar), url],
        capture_output=True, stdin=subprocess.DEVNULL,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl json failed ({r.returncode}) {url}")
    return json.loads(r.stdout.decode())


def curl_download(url: str, cookie_jar: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["curl", "-f", "-k", "-s", "-H", "Expect:", "--cookie", str(cookie_jar),
         "-o", str(dest), url],
        capture_output=True, stdin=subprocess.DEVNULL,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl download failed ({r.returncode}) {url}")


# ---------- XNAT queries ----------

def list_subjects(cookie_jar: Path):
    d = curl_json(f"{XNAT_HOST}/data/archive/projects/{PROJECT}/subjects?format=json",
                  cookie_jar)
    pat = re.compile(r"^OAS3\d+$")
    return sorted(s["label"] for s in d["ResultSet"]["Result"]
                  if pat.match(s["label"]))


def sessions_sorted(cookie_jar: Path, subject: str):
    url = (f"{XNAT_HOST}/data/archive/projects/{PROJECT}/subjects/{subject}"
           f"/experiments?format=json&xsiType=xnat:mrSessionData")
    items = curl_json(url, cookie_jar)["ResultSet"]["Result"]
    def days(s):
        m = re.search(r"_d(\d+)$", s.get("label", ""))
        return int(m.group(1)) if m else 9_999_999
    return sorted(items, key=days)


def session_has_bold(cookie_jar: Path, exp_id: str) -> bool:
    url = f"{XNAT_HOST}/data/archive/experiments/{exp_id}/scans?format=json"
    try:
        scans = curl_json(url, cookie_jar)["ResultSet"]["Result"]
    except RuntimeError:
        return False
    return any("bold" in ((s.get("type") or "") + (s.get("series_description") or "")).lower()
               for s in scans)


# ---------- per-subject pipeline ----------

def downsample_4d(nii_path: Path):
    data = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"not 4D: {data.shape}")
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)
    return F.interpolate(vol, size=TARGET_SHAPE, mode="trilinear",
                         align_corners=False).squeeze(0)


def longest_rest_and_tr(extract_dir: Path):
    """Among extracted func dirs, return (nii_path, tr) of the longest rest run."""
    best = None  # (T, nii, tr)
    for nii in extract_dir.rglob("*task-rest*_bold.nii.gz"):
        try:
            T = nib.load(str(nii)).shape[3]
        except Exception:
            continue
        tr = None
        js = Path(str(nii).replace(".nii.gz", ".json"))
        if js.exists():
            try:
                tr = json.load(open(js)).get("RepetitionTime")
            except Exception:
                pass
        if best is None or T > best[0]:
            best = (T, nii, tr)
    return (best[1], best[2]) if best else (None, None)


def process_subject(cookie_jar, subject, output_dir, tmp_dir, manifest_path):
    out_subj = output_dir / subject
    if list(out_subj.glob("*.pt")):
        return "skip"
    # find earliest session with a bold scan
    chosen = None
    for s in sessions_sorted(cookie_jar, subject):
        if session_has_bold(cookie_jar, s["ID"]):
            chosen = s
            break
    if chosen is None:
        return "no-bold"
    exp_label, exp_id = chosen["label"], chosen["ID"]
    m = re.search(r"_d(\d+)$", exp_label)
    day = m.group(1) if m else "0000"

    work = tmp_dir / exp_label
    zip_path = tmp_dir / f"{exp_label}.zip"
    try:
        url = (f"{XNAT_HOST}/data/archive/experiments/{exp_id}"
               f"/scans/bold/files?format=zip")
        curl_download(url, cookie_jar, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(work)
        nii, tr = longest_rest_and_tr(work)
        if nii is None:
            return "no-nii"
        ds = downsample_4d(nii)
        out_subj.mkdir(parents=True, exist_ok=True)
        torch.save(ds, out_subj / f"rest_d{day}_downsampled.pt")
        with open(manifest_path, "a") as f:
            f.write(f"{subject},{day},{tr},{ds.shape[0]}\n")
        return f"ok T={ds.shape[0]} tr={tr}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
        zip_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("XNAT_USERNAME", "danab95"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--reauth-every", type=int, default=40,
                    help="Re-auth (refresh cookie) every N subjects (JSESSION "
                         "expires after ~1h).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cookie_jar = tmp_dir / "cookies.jar"
    manifest_path = output_dir / f"oasis3_tr_manifest_shard{args.shard_id}.csv"
    if not manifest_path.exists():
        manifest_path.write_text("subject,day,tr,T\n")

    pw = get_password_encoded()
    if not curl_auth(args.username, pw, cookie_jar):
        print("ERROR: auth failed", file=sys.stderr); sys.exit(3)
    print(f"[shard {args.shard_id}/{args.n_shards}] auth OK", flush=True)

    subjects = list_subjects(cookie_jar)
    mine = subjects[args.shard_id::args.n_shards]
    print(f"[shard {args.shard_id}] {len(mine)} subjects (of {len(subjects)} total)",
          flush=True)

    n_ok = n_skip = n_no = n_err = 0
    for i, subj in enumerate(mine, 1):
        if i % args.reauth_every == 0:
            curl_auth(args.username, pw, cookie_jar)  # refresh cookie
        try:
            status = process_subject(cookie_jar, subj, output_dir, tmp_dir,
                                     manifest_path)
        except Exception as e:
            status = f"error: {e!r}"
        print(f"[shard {args.shard_id}] [{i}/{len(mine)}] {subj}: {status}",
              flush=True)
        if status.startswith("ok"): n_ok += 1
        elif status == "skip": n_skip += 1
        elif status.startswith("no"): n_no += 1
        else: n_err += 1

    print(f"[shard {args.shard_id}] done: ok={n_ok} skip={n_skip} "
          f"no-data={n_no} err={n_err}", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
