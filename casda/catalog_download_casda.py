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
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus

# Suppress harmless VOTable unit warnings from CASDA staging responses
warnings.filterwarnings("ignore", category=W50)

CATALOG_DIR = Path("../data/catalogs")
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

CAT_PATTERN = "selavy-image.i.EMU*taylor.0.restored.conv.components.xml"
BATCH_SIZE = 5

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

def download_catalogs(tiles):
    """Prompt for CASDA credentials, stage and download XML catalogs, convert to CSV."""
    username = input("CASDA (OPAL) username: ").strip()
    password = getpass.getpass("CASDA (OPAL) password: ")

    casda = Casda()
    # Newer astroquery removed `password` kwarg; inject credentials directly.
    casda.USERNAME = username
    casda._auth = (username, password)
    casda._authenticated = True

    failed = []

    for start in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[start:start + BATCH_SIZE]
        batch_nums = f"{start + 1}-{start + len(batch)}"

        # Within-batch skip for already-converted SBIDs
        batch_sbids = [sbid_from_filename(str(fn)) for fn in batch["filename"]]
        missing_mask = np.array([
            sbid is not None and not is_downloaded(sbid)
            for sbid in batch_sbids
        ])

        if not missing_mask.any():
            print(f"\nBatch {batch_nums}: all already downloaded, skipping.")
            continue

        print(f"\nStaging catalogs {batch_nums} of {len(tiles)}...")

        try:
            url_list = casda.stage_data(batch[missing_mask])
        except Exception as e:
            print(f"  ERROR staging batch {batch_nums}: {e}")
            failed.extend([batch_sbids[i] for i, m in enumerate(missing_mask) if m])
            continue

        # Keep only .xml URLs
        url_list = [u for u in url_list if u.split("?")[0].endswith(".xml")]

        if not url_list:
            print(f"  WARNING: no .xml URLs returned for batch {batch_nums}, skipping.")
            failed.extend([batch_sbids[i] for i, m in enumerate(missing_mask) if m])
            continue

        for url in url_list:
            parsed = urlparse(url)
            xml_filename = unquote(os.path.basename(parsed.path))
            sbid = sbid_from_filename(xml_filename)

            if sbid is None:
                print(f"  WARNING: could not extract SBID from {xml_filename}, skipping.")
                continue

            csv_out = csv_path_for_sbid(sbid)

            # Use a temp file so a crash mid-download doesn't leave a partial XML
            tmp_xml = Path(tempfile.mktemp(suffix=".xml", dir=CATALOG_DIR))

            try:
                print(f"  Downloading {xml_filename} (SBID {sbid})...")
                download_url(url, tmp_xml)

                print(f"  Converting to {csv_out.name}...")
                votable_to_csv(str(tmp_xml), str(csv_out))
                tmp_xml.unlink()  # remove XML after successful conversion
                print(f"  OK → {csv_out.name}")

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status == 403:
                    print(f"  SKIP SBID {sbid}: 403 Forbidden "
                          f"(Pawsey S3 InvalidAccessKeyId — report to CASDA support)")
                else:
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

    if failed:
        print(f"\n{len(failed)} SBID(s) failed (re-run to retry):")
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
