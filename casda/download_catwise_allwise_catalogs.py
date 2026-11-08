#!/usr/bin/env python
# coding: utf-8
"""
Download CatWISE-2020 and AllWISE photometry catalogs from IRSA TAP for every
EMU FITS tile in data/images/.

For each tile the script:
  1. Reads the FITS WCS header to derive the tile centre and angular size.
  2. Adds a configurable buffer (default 2°) around the tile footprint.
  3. Queries IRSA TAP (https://irsa.ipac.caltech.edu/TAP) for:
       - CatWISE-2020  (catwise_2020): W1/W2 positions and magnitudes
       - AllWISE       (allwise_p3as_psd): W1–W4 photometry
  4. Saves results as CSV in data/catalogs_wise/.

Output files per SBID:
  data/catalogs_wise/<SBID>_catwise_catalog.csv
  data/catalogs_wise/<SBID>_allwise_catalog.csv

Already-present files are skipped.

Usage (from repo root):
    python casda/download_catwise_allwise_catalogs.py [--config config/pipeline.yaml]
"""

import argparse
import glob
import os
import re
import time
from pathlib import Path

import pandas as pd
import pyvo as vo
import yaml
from astropy.io import fits


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dirs(cfg: dict):
    root = Path(__file__).resolve().parent.parent
    p = cfg.get("paths", {})
    return (
        root / p.get("emu_images_dir", "data/images"),
        root / p.get("wise_catalogs_dir", "data/catalogs_wise"),
    )


# ---------------------------------------------------------------------------
# Footprint helper
# ---------------------------------------------------------------------------

def get_tile_bbox(fits_file: str, buffer_deg: float = 2.0):
    """
    Return (ra_center, dec_center, width_deg, height_deg) for *fits_file*,
    padded by *buffer_deg* on each side.
    """
    with fits.open(fits_file) as hdul:
        hdr = hdul[0].header

    ra_center  = hdr["CRVAL1"]
    dec_center = hdr["CRVAL2"]
    width      = abs(hdr["CDELT1"] * hdr["NAXIS1"]) + buffer_deg
    height     = abs(hdr["CDELT2"] * hdr["NAXIS2"]) + buffer_deg
    return ra_center, dec_center, width, height


# ---------------------------------------------------------------------------
# TAP query helpers
# ---------------------------------------------------------------------------

IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

CATWISE_COLS = (
    "source_id,source_name,ra,dec,w1mag,w1sigm,w2mag,w2sigm"
)
ALLWISE_COLS = (
    "designation,ra,dec,sigra,sigdec,sigradec,"
    "w1mpro,w1sigmpro,w1snr,"
    "w2mpro,w2sigmpro,w2snr,"
    "w3mpro,w3sigmpro,w3snr,"
    "w4mpro,w4sigmpro,w4snr"
)


def _tap_box_query(
    service_url: str,
    table: str,
    columns: str,
    ra_center: float,
    dec_center: float,
    width: float,
    height: float,
    retries: int = 5,
    delay: int = 10,
):
    """
    Run an async TAP BOX query and return an astropy Table, or None on failure.
    """
    query = f"""
        SELECT {columns}
        FROM {table}
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            BOX('ICRS', {ra_center}, {dec_center}, {width}, {height})
        )
    """
    service = vo.dal.TAPService(service_url)
    for attempt in range(retries):
        try:
            return service.run_async(query).to_table()
        except Exception as exc:
            print(f"    Attempt {attempt + 1}/{retries} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"    Giving up after {retries} attempts.")
    return None


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(
    fits_file: str,
    output_dir: Path,
    sbid: str,
    buffer_deg: float,
    retries: int,
    delay: int,
) -> bool:
    """Download CatWISE and AllWISE catalogs for one EMU tile."""
    catwise_path = output_dir / f"{sbid}_catwise_catalog.csv"
    allwise_path = output_dir / f"{sbid}_allwise_catalog.csv"

    if catwise_path.exists() and allwise_path.exists():
        print(f"  SBID {sbid}: both catalogs already exist, skipping.")
        return True

    print(f"  SBID {sbid}: reading footprint from {Path(fits_file).name}")
    try:
        bbox = get_tile_bbox(fits_file, buffer_deg)
    except Exception as exc:
        print(f"  ERROR reading FITS header for SBID {sbid}: {exc}")
        return False

    ra_c, dec_c, w, h = bbox
    ok = True

    # --- CatWISE-2020 ---
    if not catwise_path.exists():
        print(f"  SBID {sbid}: querying CatWISE-2020 ...")
        result = _tap_box_query(
            IRSA_TAP_URL, "catwise_2020", CATWISE_COLS,
            ra_c, dec_c, w, h, retries, delay,
        )
        if result is not None:
            result.to_pandas().to_csv(catwise_path, index=False)
            print(f"    Saved → {catwise_path.name}  ({len(result)} rows)")
        else:
            print(f"    WARNING: no CatWISE data returned for SBID {sbid}.")
            ok = False
    else:
        print(f"  SBID {sbid}: CatWISE already exists, skipping.")

    # --- AllWISE ---
    if not allwise_path.exists():
        print(f"  SBID {sbid}: querying AllWISE ...")
        result = _tap_box_query(
            IRSA_TAP_URL, "allwise_p3as_psd", ALLWISE_COLS,
            ra_c, dec_c, w, h, retries, delay,
        )
        if result is not None:
            result.to_pandas().to_csv(allwise_path, index=False)
            print(f"    Saved → {allwise_path.name}  ({len(result)} rows)")
        else:
            print(f"    WARNING: no AllWISE data returned for SBID {sbid}.")
            ok = False
    else:
        print(f"  SBID {sbid}: AllWISE already exists, skipping.")

    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download CatWISE-2020 and AllWISE catalogs for EMU tiles."
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--buffer-deg", type=float, default=2.0,
        help="Sky buffer added around each tile footprint in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--retries", type=int, default=5,
        help="Number of TAP query retries on failure (default: 5)",
    )
    parser.add_argument(
        "--delay", type=int, default=10,
        help="Seconds to wait between retries (default: 10)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_dir, output_dir = get_dirs(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(glob.glob(str(images_dir / "*.fits")))
    if not fits_files:
        print(f"No FITS files found in {images_dir}")
        return

    print(f"Found {len(fits_files)} EMU tile(s) in {images_dir}")
    print(f"Output dir: {output_dir}")

    failed = []
    for fits_file in fits_files:
        m = re.search(r"SB(\d+)", fits_file)
        if not m:
            print(f"  WARNING: could not extract SBID from {fits_file}, skipping.")
            continue
        sbid = m.group(1)

        ok = process_tile(
            fits_file=fits_file,
            output_dir=output_dir,
            sbid=sbid,
            buffer_deg=args.buffer_deg,
            retries=args.retries,
            delay=args.delay,
        )
        if not ok:
            failed.append(sbid)

    if failed:
        print(f"\n{len(failed)} SBID(s) failed (re-run to retry):")
        for s in failed:
            print(f"  SBID {s}")
    else:
        print("\nAll tiles processed successfully.")


if __name__ == "__main__":
    main()
