"""Discover the OASIS-3 CLINICAL data structure on NITRC-IR (CDR / diagnosis),
so we can then write the label extractor for Brain-JEPA's "AD Conversion" task.

OASIS-3 stores clinical assessments (ADRC Clinical Data with CDR, dx, MMSE) as
XNAT experiments alongside the MR sessions. Their xsiType and field names are
OASIS-specific, so this script just AUTHS and dumps, for a few subjects:
  1. every experiment (xsiType + label) -> reveals the clinical experiment type
  2. the full field set of one clinical experiment -> reveals CDR/dx field names

Reuses the same curl-based XNAT auth as the scan downloader (NITRC blocks
python-requests; password from env XNAT_PASSWORD or ~/.xnat_password, URL-encoded).

Usage (on Moriah, needs XNAT creds):
    XNAT_USERNAME=danab95 python tasks/data_prep/fetch_oasis_labels/explore_oasis_clinical.py
    # or specify subjects:  ... explore_oasis_clinical.py OAS30001 OAS30002
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

XNAT_HOST = "https://www.nitrc.org/ir"
PROJECT = "OASIS3"


def get_password_encoded() -> str:
    pw = os.environ.get("XNAT_PASSWORD")
    if not pw:
        pw_file = Path("~/.xnat_password").expanduser()
        if not pw_file.exists():
            sys.exit("ERROR: no XNAT password (set XNAT_PASSWORD or ~/.xnat_password)")
        pw = pw_file.read_text().strip()
    return urllib.parse.quote(pw, safe="")


def curl_auth(username, pw_enc, cookie_jar) -> bool:
    return subprocess.run(
        ["curl", "-f", "-k", "-s", "-u", f"{username}:{pw_enc}",
         "--cookie-jar", str(cookie_jar), f"{XNAT_HOST}/data/JSESSION"],
        capture_output=True).returncode == 0


def curl_json(url, cookie_jar):
    r = subprocess.run(["curl", "-f", "-k", "-s", "--cookie", str(cookie_jar), url],
                       capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed ({r.returncode}) {url}")
    return json.loads(r.stdout.decode())


def main():
    subjects = sys.argv[1:]
    username = os.environ.get("XNAT_USERNAME", "danab95")
    pw_enc = get_password_encoded()

    with tempfile.TemporaryDirectory() as td:
        cookie = Path(td) / "cookies"
        if not curl_auth(username, pw_enc, cookie):
            sys.exit("ERROR: XNAT auth failed (check username/password)")
        print(f"auth OK as {username}\n")

        if not subjects:
            d = curl_json(f"{XNAT_HOST}/data/archive/projects/{PROJECT}/subjects?format=json", cookie)
            subjects = [s["label"] for s in d["ResultSet"]["Result"]][:3]
            print(f"no subjects given -> using first 3: {subjects}\n")

        seen_types = {}
        for subj in subjects:
            print(f"{'='*70}\nSUBJECT {subj}\n{'='*70}")
            url = (f"{XNAT_HOST}/data/archive/projects/{PROJECT}/subjects/{subj}"
                   f"/experiments?format=json")
            exps = curl_json(url, cookie)["ResultSet"]["Result"]
            for e in exps:
                xt = e.get("xsiType", "?")
                print(f"  {xt:35} label={e.get('label','?'):30} id={e.get('ID','?')}")
                seen_types.setdefault(xt, e.get("ID"))

        # dump the field set of one clinical (non-MR) experiment
        clinical = {t: i for t, i in seen_types.items() if "mrSession" not in t.lower()}
        print(f"\n{'='*70}\nDISTINCT experiment types seen: {list(seen_types)}\n"
              f"non-MR (clinical) types: {list(clinical)}\n{'='*70}")
        for xt, exp_id in list(clinical.items())[:3]:
            print(f"\n--- fields of a '{xt}' experiment (id={exp_id}) ---")
            try:
                rec = curl_json(f"{XNAT_HOST}/data/experiments/{exp_id}?format=json", cookie)
                items = rec.get("items", [])
                if items:
                    fields = items[0].get("data_fields", {})
                    for k, v in fields.items():
                        print(f"    {k:40} = {v}")
            except Exception as ex:
                print(f"    (could not fetch fields: {ex})")


if __name__ == "__main__":
    main()
