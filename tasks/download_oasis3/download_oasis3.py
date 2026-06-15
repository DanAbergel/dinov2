"""Download OASIS-3 resting-state fMRI from NITRC-IR XNAT and downsample to (45, 54, 45).

OASIS-3 (Longitudinal Multimodal Neuroimaging, Clinical, and Cognitive Dataset for
Normal Aging and Alzheimer's Disease) is hosted on the NITRC Image Repository
(NITRC-IR) at https://www.nitrc.org/ir/. Access requires being a member of the
OASIS-3 NITRC team.

Strategy: for each subject, take ONE rs-fMRI scan (the earliest MR session that
contains a resting-state BOLD scan = "baseline visit"). This gives ~1000 unique
subjects, ~85 GB after downsampling, ~12h download.

Authentication is via NITRC username + password. Password is read from one of:
  1. Env var XNAT_PASSWORD
  2. File ~/.xnat_password (chmod 600)
The password is NEVER printed, logged, or sent to anyone else.

Streams per subject:
  1. Find earliest MR session with rs-fMRI for that subject
  2. Download the resting-state scan as ZIP (~50-100 MB)
  3. Extract the .nii.gz
  4. Resample (X, Y, Z, T) -> (T, 45, 54, 45) via trilinear
  5. Save .pt, delete ZIP and extracted NIfTI

Usage:
    XNAT_PASSWORD=xxx python tasks/download_oasis3/download_oasis3.py \\
        --username danab \\
        --output-dir /sci/labs/arieljaffe/dan.abergel1/OASIS3_data/downsampled \\
        --tmp-dir /tmp/oasis3_raw
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import requests
import torch
import torch.nn.functional as F

XNAT_HOST = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"
TARGET_SHAPE = (45, 54, 45)

# Match rs-fMRI by scan "type" or "series_description" (case-insensitive).
# OASIS-3 uses labels like "rsfMRI", "Resting State BOLD", etc.
RSFMRI_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"rsfmri",
        r"resting.?state",
        r"rest.*bold",
        r"bold.*rest",
    ]
]


def get_password() -> str:
    """Read XNAT password from env var or ~/.xnat_password file."""
    pw = os.environ.get("XNAT_PASSWORD")
    if pw:
        return pw
    pw_file = Path("~/.xnat_password").expanduser()
    if pw_file.exists():
        # Refuse to read if it's world-readable (security)
        mode = pw_file.stat().st_mode & 0o777
        if mode & 0o077:
            print(f"ERROR: {pw_file} is too permissive (mode {oct(mode)}). "
                  f"Run: chmod 600 {pw_file}", file=sys.stderr)
            sys.exit(2)
        return pw_file.read_text().strip()
    print(
        "ERROR: no XNAT password found. Set XNAT_PASSWORD env var, or put it in "
        "~/.xnat_password with chmod 600.",
        file=sys.stderr,
    )
    sys.exit(2)


def authenticate(host: str, username: str, password: str) -> requests.Session:
    """Set up a session with HTTP Basic Auth applied to every request.

    NITRC-IR's /data/JSESSION endpoint can return 401 for some accounts even
    when credentials are valid for the data archive endpoints. The official
    NrgXnat bash script bypasses /JSESSION entirely and just sends Basic
    Auth on every download URL. We do the same: persistent session.auth =
    (user, pwd) means requests will send the auth header on each call.

    We sanity-check the credentials by hitting a known-public-to-team URL
    (the OASIS3 project descriptor). Failure here = real auth/access issue.
    """
    session = requests.Session()
    session.auth = (username, password)
    # Set a User-Agent — some XNAT installs are picky.
    session.headers.update({"User-Agent": "FAIR_official/download_oasis3"})

    # Sanity check: try to fetch the OASIS3 project entry.
    r = session.get(f"{host}/data/projects/{PROJECT}",
                    params={"format": "json"}, timeout=30)
    if r.status_code == 401:
        print(f"ERROR: auth failed (401) for user {username!r}. "
              f"Wrong password, or NITRC-IR account not active.", file=sys.stderr)
        sys.exit(3)
    if r.status_code == 403:
        print(f"ERROR: auth OK but no access to project {PROJECT} (403). "
              f"Contact oasisadmin to confirm project membership.", file=sys.stderr)
        sys.exit(3)
    r.raise_for_status()
    print(f"Authenticated as {username}. Project {PROJECT} accessible.")
    return session


def list_subjects(session: requests.Session) -> list:
    """List subject labels in OASIS-3 (e.g. ['OAS30001', 'OAS30002', ...])."""
    r = session.get(
        f"{XNAT_HOST}/data/projects/{PROJECT}/subjects",
        params={"format": "json"},
        timeout=60,
    )
    r.raise_for_status()
    return sorted(s["label"] for s in r.json()["ResultSet"]["Result"])


def list_mr_sessions(session: requests.Session, subject: str) -> list:
    """List MR sessions for one subject, sorted by days-from-entry (earliest first).

    Returns list of dicts with at least 'label' (e.g. 'OAS30001_MR_d0129') and
    'ID' (the XNAT-internal experiment ID).
    """
    r = session.get(
        f"{XNAT_HOST}/data/projects/{PROJECT}/subjects/{subject}/experiments",
        params={"format": "json", "xsiType": "xnat:mrSessionData"},
        timeout=60,
    )
    r.raise_for_status()
    sessions = r.json()["ResultSet"]["Result"]
    # Sort by the 'd<days>' suffix in the label.
    def days_from_label(s):
        m = re.search(r"_d(\d+)$", s.get("label", ""))
        return int(m.group(1)) if m else 9999999
    return sorted(sessions, key=days_from_label)


def list_scans(session: requests.Session, experiment_id: str) -> list:
    """List scans within an MR session.

    Returns list of dicts with 'ID', 'type', 'series_description'.
    """
    r = session.get(
        f"{XNAT_HOST}/data/experiments/{experiment_id}/scans",
        params={"format": "json"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["ResultSet"]["Result"]


def is_rsfmri(scan: dict) -> bool:
    """Check if a scan dict represents a resting-state BOLD scan."""
    haystacks = [
        scan.get("type") or "",
        scan.get("series_description") or "",
    ]
    return any(p.search(h) for h in haystacks for p in RSFMRI_PATTERNS)


def find_first_rsfmri(session: requests.Session, subject: str):
    """For one subject, return (session_label, scan_id) of the earliest rs-fMRI.

    Returns None if no rs-fMRI scan found in any session.
    """
    for mr in list_mr_sessions(session, subject):
        try:
            scans = list_scans(session, mr["ID"])
        except requests.HTTPError:
            continue
        for scan in scans:
            if is_rsfmri(scan):
                return mr["label"], mr["ID"], scan["ID"]
    return None


def download_scan_zip(session: requests.Session, experiment_id: str,
                      scan_id: str, dest_zip: Path):
    """Download the NIFTI resource (or all files) for one scan as a ZIP."""
    url = (
        f"{XNAT_HOST}/data/experiments/{experiment_id}/scans/{scan_id}"
        f"/resources/NIFTI/files"
    )
    r = session.get(url, params={"format": "zip"}, stream=True, timeout=600)
    if r.status_code == 404:
        # Fall back to all files (the resource might be named differently)
        url = (
            f"{XNAT_HOST}/data/experiments/{experiment_id}/scans/{scan_id}/files"
        )
        r = session.get(url, params={"format": "zip"}, stream=True, timeout=600)
    r.raise_for_status()
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)


def extract_first_nii_gz(zip_path: Path, dest_dir: Path) -> Path | None:
    """Extract the .nii.gz file from a ZIP and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        nii_names = [n for n in zf.namelist() if n.endswith(".nii.gz")]
        if not nii_names:
            return None
        # Take the first (and usually only) NIfTI
        nii_name = nii_names[0]
        out_path = dest_dir / Path(nii_name).name
        with zf.open(nii_name) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return out_path


def downsample_4d(nii_path: Path) -> torch.Tensor:
    """Load 4D NIfTI, downsample spatial to TARGET_SHAPE via trilinear."""
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, X, Y, Z)
    vol_ds = F.interpolate(
        vol,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return vol_ds.squeeze(0)  # (T, 45, 54, 45)


def process_subject(http_session: requests.Session, subject: str,
                    output_dir: Path, tmp_dir: Path) -> str:
    """Download + downsample one subject's baseline rs-fMRI. Returns status string."""
    out_subject_dir = output_dir / subject
    # If any .pt already exists for this subject, skip.
    if list(out_subject_dir.glob("*.pt")):
        return "skip (already done)"

    found = find_first_rsfmri(http_session, subject)
    if found is None:
        return "no rs-fMRI found"
    session_label, experiment_id, scan_id = found

    # Extract the d<days> suffix for filename
    m = re.search(r"_d(\d+)$", session_label)
    days = m.group(1) if m else "0000"

    tmp_zip = tmp_dir / f"{subject}_{session_label}_scan{scan_id}.zip"
    tmp_extract = tmp_dir / f"{subject}_{session_label}_scan{scan_id}_ext"

    try:
        download_scan_zip(http_session, experiment_id, scan_id, tmp_zip)
        nii = extract_first_nii_gz(tmp_zip, tmp_extract)
        if nii is None:
            return "ZIP contained no .nii.gz"
        ds = downsample_4d(nii)
        out_subject_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_subject_dir / f"rest_d{days}_downsampled.pt"
        torch.save(ds, out_path)
        return f"ok  shape={tuple(ds.shape)}  -> {out_path.name}"
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True,
                    help="NITRC-IR username (XNAT login)")
    ap.add_argument("--output-dir", required=True,
                    help="Where to save downsampled .pt files")
    ap.add_argument("--tmp-dir", default="/tmp/oasis3_raw",
                    help="Temp dir for raw ZIPs / NIfTIs (cleaned up per session)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N subjects (testing)")
    args = ap.parse_args()

    password = get_password()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Host       : {XNAT_HOST}")
    print(f"Project    : {PROJECT}")
    print(f"Target     : {TARGET_SHAPE}")
    print(f"Output dir : {output_dir}")
    print(f"Tmp dir    : {tmp_dir}")

    http_session = authenticate(XNAT_HOST, args.username, password)

    print("Listing subjects ...")
    subjects = list_subjects(http_session)
    print(f"Found {len(subjects)} subjects on OASIS-3.")
    if args.limit:
        subjects = subjects[:args.limit]
        print(f"Limiting to first {len(subjects)}.")

    n_ok = n_skip = n_no = n_err = 0
    for i, subj in enumerate(subjects, 1):
        try:
            status = process_subject(http_session, subj, output_dir, tmp_dir)
        except Exception as e:
            status = f"error: {e!r}"
        print(f"[{i}/{len(subjects)}] {subj}  -- {status}", flush=True)
        if status.startswith("ok"):
            n_ok += 1
        elif status.startswith("skip"):
            n_skip += 1
        elif "no rs-fMRI" in status:
            n_no += 1
        else:
            n_err += 1

    print(f"\nSummary: ok={n_ok}  skipped={n_skip}  no rs-fMRI={n_no}  errors={n_err}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
