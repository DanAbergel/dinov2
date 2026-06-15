"""Download OASIS-3 resting-state fMRI from NITRC-IR XNAT and downsample to (45, 54, 45).

OASIS-3 is hosted on NITRC-IR (https://www.nitrc.org/ir/). NITRC applies
TLS fingerprinting that blocks the Python `requests` library even with a
spoofed curl User-Agent. The official NrgXnat bash script that works uses
curl directly — so this Python wrapper does the same: every HTTP call is
delegated to a `curl` subprocess.

Strategy: 1 rs-fMRI per subject (the EARLIEST MR session containing a
resting-state BOLD scan = baseline visit).

Password is read from one of (priority order):
  1. env var XNAT_PASSWORD
  2. file ~/.xnat_password (chmod 600)
NEVER printed or logged in plaintext.

Usage:
    sbatch tasks/download_oasis3/download_oasis3.sh
    LIMIT=5 sbatch tasks/download_oasis3/download_oasis3.sh   # test 5 subjects
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

XNAT_HOST = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"
TARGET_SHAPE = (45, 54, 45)

# Match rs-fMRI by scan "type" or "series_description" (case-insensitive).
RSFMRI_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"rsfmri",
        r"resting.?state",
        r"rest.*bold",
        r"bold.*rest",
    ]
]


def get_password() -> tuple[str, str]:
    """Read XNAT password from env var or ~/.xnat_password file.

    Returns (password, source_label).
    """
    pw = os.environ.get("XNAT_PASSWORD")
    if pw:
        return pw, "env var XNAT_PASSWORD"
    pw_file = Path("~/.xnat_password").expanduser()
    if pw_file.exists():
        mode = pw_file.stat().st_mode & 0o777
        if mode & 0o077:
            print(f"ERROR: {pw_file} is too permissive (mode {oct(mode)}). "
                  f"Run: chmod 600 {pw_file}", file=sys.stderr)
            sys.exit(2)
        return pw_file.read_text().strip(), "~/.xnat_password"
    print(
        "ERROR: no XNAT password found. Set XNAT_PASSWORD env var, or put it in "
        "~/.xnat_password with chmod 600.",
        file=sys.stderr,
    )
    sys.exit(2)


def mask_password(pw: str) -> str:
    if len(pw) <= 4:
        return f"{'*' * len(pw)!r} (len={len(pw)})"
    return f"{pw[:2] + '...' + pw[-2:]!r} (len={len(pw)})"


# -------- curl wrappers -----------------------------------------------------

def curl_auth(username: str, password: str, cookie_jar: Path) -> None:
    """Authenticate against NITRC-IR; save session cookies to `cookie_jar`.

    Verbose mode (-v) is enabled so we see exactly what curl sends/receives,
    matching the diagnostic visibility of the bash script when not in
    interactive mode. -S keeps progress hidden but DOES show errors.
    """
    cmd = [
        "curl", "-f", "-k", "-S", "-v",          # -S show errors, -v verbose
        "-u", f"{username}:{password}",
        "--cookie-jar", str(cookie_jar),
        f"{XNAT_HOST}/data/JSESSION",
    ]
    # Diagnostic: print the redacted command so the user can compare with
    # what worked in their interactive shell.
    masked = [a if a != f"{username}:{password}" else f"{username}:<REDACTED>"
              for a in cmd]
    print(f"  curl cmd: {' '.join(masked)}")

    r = subprocess.run(cmd, capture_output=True)
    out = r.stdout.decode(errors="replace").strip()
    err = r.stderr.decode(errors="replace").strip()
    if r.returncode != 0:
        print(f"ERROR: curl auth failed (exit {r.returncode}).", file=sys.stderr)
        print(f"  --- curl stdout ---", file=sys.stderr)
        print(out or "(empty)", file=sys.stderr)
        print(f"  --- curl stderr ---", file=sys.stderr)
        print(err or "(empty)", file=sys.stderr)
        sys.exit(3)


def curl_get_json(url: str, cookie_jar: Path) -> dict | list:
    """GET a URL with the saved cookie and parse the JSON response."""
    cmd = [
        "curl", "-f", "-k", "-s",
        "--cookie", str(cookie_jar),
        url,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"curl GET failed ({r.returncode}) on {url}: "
            f"{r.stderr.decode(errors='replace')}"
        )
    try:
        return json.loads(r.stdout.decode())
    except json.JSONDecodeError:
        snippet = r.stdout[:200].decode(errors="replace")
        raise RuntimeError(f"non-JSON response from {url}: {snippet!r}")


def curl_download(url: str, cookie_jar: Path, dest: Path) -> None:
    """Download a binary file (e.g. a ZIP) with the saved cookie."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-f", "-k", "-s",
        "--cookie", str(cookie_jar),
        "-o", str(dest),
        url,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"curl download failed ({r.returncode}) on {url}: "
            f"{r.stderr.decode(errors='replace')}"
        )


# -------- XNAT API wrappers (built on curl) ---------------------------------

def list_subjects(cookie_jar: Path) -> list[str]:
    url = f"{XNAT_HOST}/data/archive/projects/{PROJECT}/subjects?format=json"
    data = curl_get_json(url, cookie_jar)
    return sorted(s["label"] for s in data["ResultSet"]["Result"])


def list_mr_sessions(cookie_jar: Path, subject: str) -> list[dict]:
    url = (
        f"{XNAT_HOST}/data/archive/projects/{PROJECT}/subjects/{subject}"
        f"/experiments?format=json&xsiType=xnat:mrSessionData"
    )
    data = curl_get_json(url, cookie_jar)
    sessions = data["ResultSet"]["Result"]
    def days(s):
        m = re.search(r"_d(\d+)$", s.get("label", ""))
        return int(m.group(1)) if m else 9_999_999
    return sorted(sessions, key=days)


def list_scans(cookie_jar: Path, experiment_id: str) -> list[dict]:
    url = (
        f"{XNAT_HOST}/data/archive/experiments/{experiment_id}/scans?format=json"
    )
    data = curl_get_json(url, cookie_jar)
    return data["ResultSet"]["Result"]


def is_rsfmri(scan: dict) -> bool:
    haystacks = [
        scan.get("type") or "",
        scan.get("series_description") or "",
    ]
    return any(p.search(h) for h in haystacks for p in RSFMRI_PATTERNS)


def find_first_rsfmri(cookie_jar: Path, subject: str):
    """Return (session_label, experiment_id, scan_id) for earliest rs-fMRI.

    None if not found.
    """
    for mr in list_mr_sessions(cookie_jar, subject):
        try:
            scans = list_scans(cookie_jar, mr["ID"])
        except RuntimeError:
            continue
        for sc in scans:
            if is_rsfmri(sc):
                return mr["label"], mr["ID"], sc["ID"]
    return None


def download_scan_zip(cookie_jar: Path, experiment_id: str,
                      scan_id: str, dest_zip: Path):
    """Download the scan's NIfTI resource (or fallback to all files) as ZIP."""
    url = (
        f"{XNAT_HOST}/data/archive/experiments/{experiment_id}/scans/{scan_id}"
        f"/resources/NIFTI/files?format=zip"
    )
    try:
        curl_download(url, cookie_jar, dest_zip)
    except RuntimeError:
        # Fallback: all files for the scan
        url = (
            f"{XNAT_HOST}/data/archive/experiments/{experiment_id}/scans/{scan_id}"
            f"/files?format=zip"
        )
        curl_download(url, cookie_jar, dest_zip)


# -------- ZIP + downsample --------------------------------------------------

def extract_first_nii_gz(zip_path: Path, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        nii = [n for n in zf.namelist() if n.endswith(".nii.gz")]
        if not nii:
            return None
        out_path = dest_dir / Path(nii[0]).name
        with zf.open(nii[0]) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return out_path


def downsample_4d(nii_path: Path) -> torch.Tensor:
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    vol = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)
    vol_ds = F.interpolate(
        vol,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return vol_ds.squeeze(0)


def process_subject(cookie_jar: Path, subject: str,
                    output_dir: Path, tmp_dir: Path) -> str:
    out_subj_dir = output_dir / subject
    if list(out_subj_dir.glob("*.pt")):
        return "skip (already done)"
    found = find_first_rsfmri(cookie_jar, subject)
    if found is None:
        return "no rs-fMRI found"
    session_label, exp_id, scan_id = found
    m = re.search(r"_d(\d+)$", session_label)
    days = m.group(1) if m else "0000"
    tmp_zip = tmp_dir / f"{subject}_{session_label}_scan{scan_id}.zip"
    tmp_extract = tmp_dir / f"{subject}_{session_label}_scan{scan_id}_ext"
    try:
        download_scan_zip(cookie_jar, exp_id, scan_id, tmp_zip)
        nii = extract_first_nii_gz(tmp_zip, tmp_extract)
        if nii is None:
            return "ZIP contained no .nii.gz"
        ds = downsample_4d(nii)
        out_subj_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_subj_dir / f"rest_d{days}_downsampled.pt"
        torch.save(ds, out_path)
        return f"ok  shape={tuple(ds.shape)}  -> {out_path.name}"
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)


# -------- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/oasis3_raw")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    password, pw_source = get_password()
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Host       : {XNAT_HOST}")
    print(f"Project    : {PROJECT}")
    print(f"Target     : {TARGET_SHAPE}")
    print(f"Output dir : {output_dir}")
    print(f"Tmp dir    : {tmp_dir}")
    print(f"Username   : {args.username}")
    print(f"Password   : {mask_password(password)}  source: {pw_source}")

    # Auth: save cookie to a private temp file (chmod 600).
    cookie_jar = Path(tempfile.mkstemp(prefix="oasis3_cookie_", suffix=".jar")[1])
    cookie_jar.chmod(0o600)
    try:
        print("Authenticating via curl subprocess ...")
        curl_auth(args.username, password, cookie_jar)
        print(f"Authenticated. Cookie jar: {cookie_jar}")

        print("Listing subjects ...")
        subjects = list_subjects(cookie_jar)
        print(f"Found {len(subjects)} subjects on OASIS-3.")
        if args.limit:
            subjects = subjects[:args.limit]
            print(f"Limiting to first {len(subjects)}.")

        n_ok = n_skip = n_no = n_err = 0
        for i, subj in enumerate(subjects, 1):
            try:
                status = process_subject(cookie_jar, subj, output_dir, tmp_dir)
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

        print(f"\nSummary: ok={n_ok} skipped={n_skip} no rs-fMRI={n_no} errors={n_err}")
    finally:
        if cookie_jar.exists():
            cookie_jar.unlink()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
