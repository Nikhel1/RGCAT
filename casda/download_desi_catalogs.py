#!/usr/bin/env python
# coding: utf-8
"""
Download DESI DR10 photometry and photometric-redshift catalogs from NOIRLab
DataLab TAP for every EMU FITS tile in data/images/.

For each tile the script:
  1. Reads the FITS WCS header to derive the tile centre and angular size.
  2. Adds a configurable buffer (default 2°) around the tile footprint.
  3. Queries NOIRLab DataLab TAP (https://datalab.noirlab.edu/tap) for:
       - DESI DR10 photometry (ls_dr10.tractor):
           ls_id, ra, dec, type, pmra, pmdec, parallax,
           mag_g/r/i/z, mag_w1/w2/w3/w4
           filtered to 15 < mag_r < 23 and mag_z < 22
       - DESI DR10 photo-z (ls_dr10.photo_z):
           z_phot_median, z_phot_median_i, z_phot_std_i, z_spec
           queried by ls_id in batches
  4. Saves results as CSV in data/catalogs_desi/.

Output files per SBID:
  data/catalogs_desi/<SBID>_desi_dr10_catalog.csv
  data/catalogs_desi/<SBID>_desi_dr10_z_catalog.csv

Already-present files are skipped.

Usage (from repo root):
    python casda/download_desi_catalogs.py [--config config/pipeline.yaml]
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
        root / p.get("desi_catalogs_dir", "data/catalogs_desi"),
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

NOIRLAB_TAP_URL = "https://datalab.noirlab.edu/tap"

DESI_PHOT_COLS = (
    "ls_id, ra, dec, type, pmra, pmdec, parallax, "
    "mag_g, mag_r, mag_i, mag_z, mag_w1, mag_w2, mag_w3, mag_w4"
)
DESI_PHOTOZ_COLS = (
    "ls_id, z_phot_median, z_phot_median_i, z_phot_std_i, z_spec"
)


def query_desi_photometry(
    ra_center: float,
    dec_center: float,
    width: float,
    height: float,
    retries: int = 5,
    delay: int = 10,
):
    """
    Query DESI DR10 photometry (ls_dr10.tractor) and return an astropy Table,
    or None on failure.
    """
    ra_min  = ra_center - width  / 2
    ra_max  = ra_center + width  / 2
    dec_min = dec_center - height / 2
    dec_max = dec_center + height / 2

    query = f"""
        SELECT {DESI_PHOT_COLS}
        FROM ls_dr10.tractor
        WHERE ra  > ({ra_min})  AND ra  < ({ra_max})
          AND dec > ({dec_min}) AND dec < ({dec_max})
          AND mag_r > 15 AND mag_r < 23
          AND mag_z < 22
    """
    service = vo.dal.TAPService(NOIRLAB_TAP_URL)
    for attempt in range(retries):
        try:
            return service.run_async(query).to_table()
        except Exception as exc:
            print(f"    Attempt {attempt + 1}/{retries} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"    Giving up after {retries} attempts.")
    return None


def query_desi_photoz(
    ls_ids,
    retries: int = 5,
    delay: int = 10,
    batch_size: int = 10_000,
):
    """
    Query DESI DR10 photo-z (ls_dr10.photo_z) for the given ls_id array in
    batches of *batch_size*.  Returns a pandas DataFrame or None on failure.
    """
    id_strs = [str(i) for i in ls_ids]
    service = vo.dal.TAPService(NOIRLAB_TAP_URL)
    parts = []

    for batch_start in range(0, len(id_strs), batch_size):
        batch = id_strs[batch_start:batch_start + batch_size]
        id_list = ",".join(batch)
        query = f"""
            SELECT {DESI_PHOTOZ_COLS}
            FROM ls_dr10.photo_z
            WHERE ls_id IN ({id_list})
        """
        for attempt in range(retries):
            try:
                df = service.run_async(query).to_table().to_pandas()
                parts.append(df)
                break
            except Exception as exc:
                print(
                    f"    Batch {batch_start // batch_size + 1}: "
                    f"attempt {attempt + 1}/{retries} failed: {exc}"
                )
                if attempt < retries - 1:
                    time.sleep(delay)
        else:
            print(f"    Giving up on batch starting at index {batch_start}.")
            return None

    if parts:
        return pd.concat(parts, ignore_index=True)
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
    """Download DESI DR10 photometry and photo-z catalogs for one EMU tile."""
    phot_path   = output_dir / f"{sbid}_desi_dr10_catalog.csv"
    photoz_path = output_dir / f"{sbid}_desi_dr10_z_catalog.csv"

    if phot_path.exists() and photoz_path.exists():
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

    # --- DESI DR10 photometry ---
    if not phot_path.exists():
        print(f"  SBID {sbid}: querying DESI DR10 photometry ...")
        result = query_desi_photometry(ra_c, dec_c, w, h, retries, delay)
        if result is not None:
            phot_df = result.to_pandas()
            phot_df.to_csv(phot_path, index=False)
            print(f"    Saved → {phot_path.name}  ({len(phot_df)} rows)")
        else:
            print(f"    WARNING: no DESI DR10 photometry returned for SBID {sbid}.")
            return False  # can't do photo-z without photometry ls_ids
    else:
        print(f"  SBID {sbid}: DESI DR10 photometry already exists, loading ls_ids.")
        phot_df = pd.read_csv(phot_path)

    # --- DESI DR10 photo-z ---
    if not photoz_path.exists():
        if phot_df.empty:
            print(f"  SBID {sbid}: photometry table is empty, skipping photo-z.")
            return True

        print(f"  SBID {sbid}: querying DESI DR10 photo-z "
              f"({len(phot_df)} ls_ids) ...")
        photoz_df = query_desi_photoz(
            phot_df["ls_id"].values, retries, delay
        )
        if photoz_df is not None:
            photoz_df.to_csv(photoz_path, index=False)
            print(f"    Saved → {photoz_path.name}  ({len(photoz_df)} rows)")
        else:
            print(f"    WARNING: no DESI photo-z data returned for SBID {sbid}.")
            ok = False
    else:
        print(f"  SBID {sbid}: DESI photo-z already exists, skipping.")

    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download DESI DR10 photometry and photo-z catalogs for EMU tiles."
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
    parser.add_argument(
        "--batch-size", type=int, default=10_000,
        help="ls_id batch size for photo-z queries (default: 10000)",
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
