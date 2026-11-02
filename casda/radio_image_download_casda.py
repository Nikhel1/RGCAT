#!/usr/bin/env python
# coding: utf-8

import fnmatch
import getpass
import os
import warnings
from pathlib import Path
from urllib.parse import urlparse, unquote

import numpy as np
import requests
from astropy.io.votable.exceptions import W50
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus

# Suppress harmless VOTable unit warnings from CASDA's staging responses
warnings.filterwarnings("ignore", category=W50)

IMAGE_DIR = Path("../data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = "image.i.EMU*taylor.0.restored.conv.fits"
BATCH_SIZE = 5  # tiles staged/downloaded per request

casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")


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


def download_url(url, savedir):
    """
    Download a single pre-signed URL to savedir.

    Returns the local filepath on success, raises on error.
    Pre-signed S3 URLs are self-authenticating; no auth header needed.
    Supports resuming partial downloads via HTTP Range requests.
    """
    parsed = urlparse(url)
    filename = unquote(os.path.basename(parsed.path))
    local_path = Path(savedir) / filename

    # Determine how many bytes we already have (partial download resume)
    existing = local_path.stat().st_size if local_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        # 206 = partial content (resume), 200 = full file
        if resp.status_code == 416:
            # Range not satisfiable — file already complete
            return str(local_path)
        resp.raise_for_status()

        mode = "ab" if existing > 0 and resp.status_code == 206 else "wb"
        total = int(resp.headers.get("content-length", 0)) + (existing if mode == "ab" else 0)

        downloaded = existing
        with open(local_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {filename}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB "
                          f"({pct:.1f}%)", end="", flush=True)
        print()  # newline after progress

    return str(local_path)


def download_tiles(tiles):
    """Prompt for CASDA (OPAL) credentials, then stage and download .fits files only."""
    username = input("CASDA (OPAL) username: ").strip()
    password = getpass.getpass("CASDA (OPAL) password: ")

    casda = Casda()
    # Newer astroquery dropped the `password` kwarg and uses keyring, which is
    # unavailable on headless HPC nodes.  Bypass login() entirely and inject
    # credentials directly — replicating exactly what a successful _login() does.
    casda.USERNAME = username
    casda._auth = (username, password)
    casda._authenticated = True

    failed_tiles = []

    for start in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[start:start + BATCH_SIZE]
        batch_nums = f"{start + 1}-{start + len(batch)}"

        # Skip tiles already on disk (handles re-runs after a mid-batch crash)
        batch_missing_mask = np.array([
            not is_downloaded(str(fn)) for fn in batch["filename"]
        ])
        if not batch_missing_mask.any():
            print(f"\nBatch {batch_nums}: all already downloaded, skipping.")
            continue

        print(f"\nStaging tiles {batch_nums} of {len(tiles)}...")

        try:
            url_list = casda.stage_data(batch[batch_missing_mask])
        except Exception as e:
            print(f"  ERROR staging batch {batch_nums}: {e}")
            failed_tiles.extend(list(batch["filename"][batch_missing_mask]))
            continue

        # Keep only .fits URLs (staging also returns .checksum files)
        url_list = [url for url in url_list if url.split("?")[0].endswith(".fits")]

        if not url_list:
            print(f"  WARNING: no .fits URLs returned for batch {batch_nums}, skipping.")
            failed_tiles.extend(list(batch["filename"][batch_missing_mask]))
            continue

        # Download each file individually so one 403/timeout doesn't kill the batch
        for url in url_list:
            parsed = urlparse(url)
            filename = unquote(os.path.basename(parsed.path))
            local_path = IMAGE_DIR / filename
            try:
                print(f"  Downloading {filename}...")
                download_url(url, str(IMAGE_DIR))
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status == 403:
                    print(f"  SKIP {filename}: 403 Forbidden "
                          f"(Pawsey S3 InvalidAccessKeyId — CASDA staged this URL with a "
                          f"revoked/invalid S3 key; report to CASDA support)")
                else:
                    print(f"  ERROR {filename}: HTTP {status} — {e}")
                # Remove zero-byte partial file if created
                if local_path.exists() and local_path.stat().st_size == 0:
                    local_path.unlink()
                failed_tiles.append(filename)
            except Exception as e:
                print(f"  ERROR {filename}: {e}")
                if local_path.exists() and local_path.stat().st_size == 0:
                    local_path.unlink()
                failed_tiles.append(filename)

    if failed_tiles:
        print(f"\n{len(failed_tiles)} tile(s) failed (re-run to retry):")
        for fn in failed_tiles:
            print(f"  {fn}")


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

    # Optional: estimated total size (access_estsize is in kB)
    if "access_estsize" in tiles.colnames:
        total_gb = np.sum(tiles["access_estsize"][missing_mask]) / 1e6
        print(f"Estimated download size: {total_gb:.1f} GB")

    download_tiles(tiles[missing_mask])

    # Final check
    still_missing = [fn for fn in filenames if not is_downloaded(fn)]
    if still_missing:
        print("\nThese tiles are still missing:")
        for fn in still_missing:
            print("  ", fn)
    else:
        print("\nAll tiles are now in", IMAGE_DIR)


if __name__ == "__main__":
    main()
