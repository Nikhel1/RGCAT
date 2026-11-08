#!/usr/bin/env python
# coding: utf-8
"""
Download EMU selavy component catalogues from CASDA, convert XML → CSV.

CASDA files:  selavy-image.i.EMU_*.SB<SBID>.cont.taylor.0.restored.conv.components.xml
Local output: ../data/catalogs/AS201_Continuum_Component_Catalogue_<SBID>_<SBID>.csv

Already-present SBIDs are skipped.  XML files are removed after successful conversion.
"""

import fnmatch
import getpass
import io
import os
import re
import tempfile
import warnings
from pathlib import Path
from urllib.parse import urlparse, unquote

import numpy as np
import requests
from astropy.io.votable import parse as parse_votable
from astropy.io.votable.exceptions import W50
from astroquery.utils.tap.core import TapPlus

# Suppress harmless VOTable unit warnings from CASDA staging responses
warnings.filterwarnings("ignore", category=W50)

CATALOG_DIR = Path("../data/catalogs")
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

CAT_PATTERN = "selavy-image.i.EMU*taylor.0.restored.conv.components.xml"

casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sbid_from_filename(filename):
    """Extract numeric SBID from a selavy filename like ...SB12345..."""
    m = re.search(r"SB(\d+)", filename)
    return m.group(1) if m else None


def csv_path_for_sbid(sbid):
    return CATALOG_DIR / f"AS201_Continuum_Component_Catalogue_{sbid}_{sbid}.csv"


def is_downloaded(sbid):
    p = csv_path_for_sbid(sbid)
    return p.exists() and p.stat().st_size > 0


def votable_to_csv(xml_path, csv_path):
    """Parse a VOTable XML file and write it as CSV."""
    votable = parse_votable(xml_path)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    df = table.to_pandas()
    df.to_csv(csv_path, index=False)


def download_url(url, dest_path):
    """
    Download a pre-signed S3 URL to dest_path with resume support.
    Returns dest_path on success, raises on error.
    """
    existing = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        if resp.status_code == 416:
            return dest_path  # already complete
        resp.raise_for_status()

        mode = "ab" if existing > 0 and resp.status_code == 206 else "wb"
        total = int(resp.headers.get("content-length", 0)) + (existing if mode == "ab" else 0)
        downloaded = existing

        with open(dest_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    name = dest_path.name
                    print(f"\r  {name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)",
                          end="", flush=True)
        print()

    return dest_path


# ---------------------------------------------------------------------------
# CASDA query
# ---------------------------------------------------------------------------

def list_casda_catalogs():
    """Return astropy Table of all EMU selavy component catalogs in CASDA."""
    sql_pattern = CAT_PATTERN.replace("*", "%")
    query = f"""
        SELECT *
        FROM ivoa.obscore
        WHERE filename LIKE '{sql_pattern}'
    """
    job = casdatap.launch_job_async(query)
    tiles = job.get_results()

    # Exact pattern match
    mask = [fnmatch.fnmatch(str(fn), CAT_PATTERN) for fn in tiles["filename"]]
    tiles = tiles[mask]

    # Drop duplicates by SBID (keep first)
    sbids = [sbid_from_filename(str(fn)) for fn in tiles["filename"]]
    seen, keep = set(), []
    for i, sbid in enumerate(sbids):
        if sbid and sbid not in seen:
            seen.add(sbid)
            keep.append(i)
    return tiles[keep]


# ---------------------------------------------------------------------------
# Download + convert
# ---------------------------------------------------------------------------

def resolve_datalink(access_url, auth):
    """
    Resolve a CASDA DataLink URL to the direct file download URL.

    The DataLink VOTable contains a results table with rows.  For public (or
    accessible) files one of the rows has:
      - semantics == '#this'  (the primary file)
      - access_url            (the direct S3 / HTTP download URL)

    Falls back to the first row that has a non-empty access_url if '#this'
    is not present.

    Returns (url, error_message) where url is the download URL or None,
    and error_message is the CASDA error string if access was denied.
    """
    resp = requests.get(access_url, auth=auth, timeout=60)
    resp.raise_for_status()

    vot = parse_votable(io.BytesIO(resp.content), verify="warn")
    for resource in vot.resources:
        if resource.type == "results":
            table = resource.tables[0].array
            # Check for an error row first
            for row in table:
                sem = row["semantics"]
                if isinstance(sem, bytes):
                    sem = sem.decode()
                if sem.strip() == "#error":
                    err = row["error_message"]
                    if isinstance(err, bytes):
                        err = err.decode()
                    return None, err.strip()

            # Prefer the '#this' semantic row (primary file)
            for row in table:
                sem = row["semantics"]
                if isinstance(sem, bytes):
                    sem = sem.decode()
                url = row["access_url"]
                if isinstance(url, bytes):
                    url = url.decode()
                if sem.strip() == "#this" and url.strip():
                    return url.strip(), None

            # Fallback: any row with a non-empty access_url
            for row in table:
                url = row["access_url"]
                if isinstance(url, bytes):
                    url = url.decode()
                if url.strip():
                    return url.strip(), None

    return None, "No download URL found in DataLink response"


def download_catalogs(tiles):
    """
    Resolve DataLink URLs and download XML catalogs directly, then convert to CSV.

    CASDA public catalogues are accessible via their DataLink access_url with
    OPAL credentials.  The DataLink response is a VOTable that contains the
    direct S3 download URL — no async staging job is needed.
    """
    username = input("CASDA (OPAL) username: ").strip()
    password = getpass.getpass("CASDA (OPAL) password: ")
    auth = (username, password)

    # Verify credentials before starting
    login_resp = requests.get(
        "https://data.csiro.au/casda_vo_proxy/vo/tap/availability",
        auth=auth, timeout=30
    )
    if login_resp.status_code != 200:
        raise RuntimeError(
            f"CASDA authentication failed (HTTP {login_resp.status_code}). "
            "Check your OPAL username/password at https://opal.atnf.csiro.au/"
        )
    print("Authentication successful.")

    failed = []
    denied = []

    for i, row in enumerate(tiles, start=1):
        sbid = sbid_from_filename(str(row["filename"]))
        if sbid is None:
            print(f"  WARNING [{i}/{len(tiles)}]: could not extract SBID from "
                  f"{row['filename']}, skipping.")
            continue

        if is_downloaded(sbid):
            print(f"[{i}/{len(tiles)}] SBID {sbid}: already downloaded, skipping.")
            continue

        access_url = str(row["access_url"])
        csv_out = csv_path_for_sbid(sbid)
        tmp_xml = Path(tempfile.mktemp(suffix=".xml", dir=CATALOG_DIR))

        print(f"[{i}/{len(tiles)}] SBID {sbid}: resolving DataLink...", end=" ", flush=True)
        try:
            dl_url, dl_err = resolve_datalink(access_url, auth)
        except Exception as e:
            print(f"\n  ERROR resolving DataLink for SBID {sbid}: {e}")
            failed.append(sbid)
            continue

        if not dl_url:
            if "permission" in (dl_err or "").lower() or "access" in (dl_err or "").lower():
                print(f"\n  SKIP (embargoed): {dl_err}")
                denied.append(sbid)
            else:
                print(f"\n  SKIP: {dl_err}")
                failed.append(sbid)
            continue

        xml_filename = unquote(os.path.basename(urlparse(dl_url).path))
        print(f"downloading {xml_filename}...")

        try:
            download_url(dl_url, tmp_xml)

            print(f"  Converting to {csv_out.name}...")
            votable_to_csv(str(tmp_xml), str(csv_out))
            tmp_xml.unlink()
            print(f"  OK → {csv_out.name}")

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"  ERROR SBID {sbid}: HTTP {status} — {e}")
            if tmp_xml.exists():
                tmp_xml.unlink()
            failed.append(sbid)

        except Exception as e:
            print(f"  ERROR SBID {sbid}: {e}")
            if tmp_xml.exists():
                tmp_xml.unlink()
            if csv_out.exists() and csv_out.stat().st_size == 0:
                csv_out.unlink()
            failed.append(sbid)

    if denied:
        print(f"\n{len(denied)} SBID(s) skipped (embargoed / no permission — re-run won't help):")
        for sbid in denied:
            print(f"  SBID {sbid}")

    if failed:
        print(f"\n{len(failed)} SBID(s) failed with errors (re-run to retry):")
        for sbid in failed:
            print(f"  SBID {sbid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Querying CASDA for EMU component catalogs...")
    tiles = list_casda_catalogs()
    print(f"Found {len(tiles)} catalog(s) in CASDA.")

    sbids = [sbid_from_filename(str(fn)) for fn in tiles["filename"]]
    have = [s for s in sbids if s and is_downloaded(s)]
    missing_mask = np.array([s is not None and not is_downloaded(s) for s in sbids])

    print(f"Already in {CATALOG_DIR}: {len(have)}")
    print(f"To download: {int(missing_mask.sum())}")

    if not missing_mask.any():
        print("Nothing to download, all catalogs are already local.")
        return

    download_catalogs(tiles[missing_mask])

    # Final check
    still_missing = [s for s in sbids if s and not is_downloaded(s)]
    if still_missing:
        print(f"\n{len(still_missing)} catalog(s) still missing:")
        for s in still_missing:
            print(f"  SBID {s}")
    else:
        print(f"\nAll catalogs are now in {CATALOG_DIR}")


if __name__ == "__main__":
    main()
