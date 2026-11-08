#!/usr/bin/env python
# coding: utf-8
"""
Download EMU radio continuum FITS tiles from CASDA.

CASDA files:  image.i.EMU_*.SB<SBID>.cont.taylor.0.restored.conv.fits
Local output: ../data/images/<filename>

Already-present files are skipped.  Uses DataLink direct download (no async
staging job) so it works for both public and project-accessible tiles.
"""

import fnmatch
import getpass
import io
import os
import warnings
from pathlib import Path
from urllib.parse import urlparse, unquote

import numpy as np
import requests
from astropy.io.votable import parse as parse_votable
from astropy.io.votable.exceptions import W50
from astroquery.utils.tap.core import TapPlus

# Suppress harmless VOTable unit warnings
warnings.filterwarnings("ignore", category=W50)

IMAGE_DIR = Path("../data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = "image.i.EMU*taylor.0.restored.conv.fits"

casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")


# ---------------------------------------------------------------------------
# CASDA query
# ---------------------------------------------------------------------------

def list_casda_tiles():
    """Return an astropy Table of all CASDA images matching PATTERN."""
    sql_pattern = PATTERN.replace("*", "%")
    query = f"""
        SELECT *
        FROM ivoa.obscore
        WHERE filename LIKE '{sql_pattern}'
    """
    job = casdatap.launch_job_async(query)
    tiles = job.get_results()

    # Safety check: keep only exact pattern matches
    mask = [fnmatch.fnmatch(fn, PATTERN) for fn in tiles["filename"]]
    tiles = tiles[mask]

    # Drop duplicate filenames (keep first occurrence)
    _, unique_idx = np.unique(np.array(tiles["filename"]), return_index=True)
    return tiles[np.sort(unique_idx)]


def is_downloaded(filename):
    path = IMAGE_DIR / filename
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# DataLink resolution
# ---------------------------------------------------------------------------

def resolve_datalink(access_url, auth):
    """
    Resolve a CASDA DataLink URL to the direct file download URL.

    Returns (url, error_message):
      - url is the direct download URL, or None on failure
      - error_message is set if CASDA denied access or no URL was found
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


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_url(url, dest_path):
    """
    Download url to dest_path with resume support.
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
                    print(f"\r  {dest_path.name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB "
                          f"({pct:.1f}%)", end="", flush=True)
        print()

    return dest_path


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def download_tiles(tiles):
    """
    Resolve DataLink URLs and download FITS tiles directly.

    For each tile the DataLink endpoint is queried with OPAL credentials to
    obtain a pre-signed direct download URL.  No async staging job is needed.
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
        filename = str(row["filename"])

        if is_downloaded(filename):
            print(f"[{i}/{len(tiles)}] {filename}: already downloaded, skipping.")
            continue

        access_url = str(row["access_url"])
        local_path = IMAGE_DIR / filename

        print(f"[{i}/{len(tiles)}] {filename}: resolving DataLink...", end=" ", flush=True)
        try:
            dl_url, dl_err = resolve_datalink(access_url, auth)
        except Exception as e:
            print(f"\n  ERROR resolving DataLink: {e}")
            failed.append(filename)
            continue

        if not dl_url:
            if "permission" in (dl_err or "").lower() or "access" in (dl_err or "").lower():
                print(f"\n  SKIP (embargoed): {dl_err}")
                denied.append(filename)
            else:
                print(f"\n  SKIP: {dl_err}")
                failed.append(filename)
            continue

        print("downloading...")
        try:
            download_url(dl_url, local_path)
            print(f"  OK → {filename}")

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status == 403:
                # Pawsey S3 InvalidAccessKeyId — the pre-signed URL CASDA gave us
                # has an expired/invalid S3 key.  This is a CASDA-side issue;
                # retrying will get a fresh URL from DataLink.
                print(f"  ERROR {filename}: 403 Forbidden "
                      f"(Pawsey S3 bad key — re-run to get a fresh URL from CASDA)")
            else:
                print(f"  ERROR {filename}: HTTP {status} — {e}")
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink()
            failed.append(filename)

        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink()
            failed.append(filename)

    if denied:
        print(f"\n{len(denied)} tile(s) skipped (embargoed / no permission — re-run won't help):")
        for fn in denied:
            print(f"  {fn}")

    if failed:
        print(f"\n{len(failed)} tile(s) failed with errors (re-run to retry):")
        for fn in failed:
            print(f"  {fn}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Querying CASDA for tiles matching", PATTERN)
    tiles = list_casda_tiles()
    print(f"Found {len(tiles)} tile(s) in CASDA.")

    filenames = [str(fn) for fn in tiles["filename"]]
    have = [fn for fn in filenames if is_downloaded(fn)]
    missing_mask = np.array([not is_downloaded(fn) for fn in filenames])

    print(f"Already in {IMAGE_DIR}: {len(have)}")
    print(f"To download: {int(missing_mask.sum())}")

    if not missing_mask.any():
        print("Nothing to download, all tiles are already local.")
        return

    if "access_estsize" in tiles.colnames:
        total_gb = np.sum(tiles["access_estsize"][missing_mask]) / 1e6
        print(f"Estimated download size: {total_gb:.1f} GB")

    download_tiles(tiles[missing_mask])

    still_missing = [fn for fn in filenames if not is_downloaded(fn)]
    if still_missing:
        print(f"\n{len(still_missing)} tile(s) still missing after run.")
    else:
        print("\nAll tiles are now in", IMAGE_DIR)


if __name__ == "__main__":
    main()
