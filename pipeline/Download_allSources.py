#!/usr/bin/env python
"""
Download_allSources.py
----------------------
For a given EMU tile (identified by its filename), downloads:
  - EMU radio cutouts  →  all_SBs/<SBid>/All_Images_8arcmV1/
  - WISE IR cutouts    →  all_SBs/<SBid>/All_Images_8arcmV1_wise/

Usage (from within CatalogNET_V1/):
    python Download_allSources.py --Image_SBid image.i.EMU_0044-46.SB50180....fits
"""

import argparse
import glob
import logging
import os
import re
import sys
import warnings

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse as parse_votable
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm import tqdm

warnings.simplefilter("ignore", category=AstropyWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Download_allSources")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download EMU + WISE cutouts for all sources in a tile."
    )
    parser.add_argument(
        "--Image_SBid",
        type=str,
        required=True,
        help="Filename of the EMU FITS tile (basename only, e.g. image.i.EMU_0044-46.SB50180....fits)",
    )
    parser.add_argument(
        "--cutout_size",
        type=int,
        default=240,
        help="Pixel size of each cutout (default: 240)",
    )
    parser.add_argument(
        "--angular_size",
        type=float,
        default=8.0,
        help="Angular size of each cutout in arcminutes (default: 8.0)",
    )
    parser.add_argument(
        "--images_base_dir",
        type=str,
        default="./EMUcatalog_main/images",
        help="Base directory that holds EMU FITS tiles",
    )
    return parser


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def votable_to_pandas(votable_file: str) -> pd.DataFrame:
    votable = parse_votable(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()


def find_catalog(catalogs_dir: str, sbid: str) -> str:
    """Return path to the CSV selavy catalog for this SBid.

    Raises FileNotFoundError with a descriptive message if zero or
    multiple matches are found.
    """
    pattern = os.path.join(catalogs_dir, f"*{sbid}*.csv")
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No catalog CSV found matching pattern: {pattern}"
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple catalog files match SBid %s — using the first: %s",
            sbid,
            matches[0],
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Cutout logic
# ---------------------------------------------------------------------------
def make_wise_cutout(
    pos: SkyCoord,
    angular_size: u.Quantity,
    cutout_size: int,
    img_wise,
    wcs_wise: WCS,
    wise_outpath: str,
    radio_outpath: str,
) -> bool:
    """
    Extract a WISE IR cutout centred on *pos*, reproject to match the radio
    cutout WCS, and write to *wise_outpath* (reprojected file has _repo suffix).

    Returns True on success, False on any failure.
    """
    extended_size = (
        int((angular_size + 2.5 * u.arcmin).to(u.arcmin).value * 60 / 2),
        int((angular_size + 2.5 * u.arcmin).to(u.arcmin).value * 60 / 2),
    )
    try:
        d_wise = Cutout2D(
            img_wise[0].data.squeeze(),
            position=pos,
            size=(cutout_size + 30, cutout_size + 30),  # slightly larger before reproject
            fill_value=np.nan,
            wcs=wcs_wise.celestial,
        )
    except Exception as exc:
        logger.debug("WISE Cutout2D failed for %s: %s", pos, exc)
        return False

    if np.sum(~np.isfinite(d_wise.data)) > 0:
        logger.debug("WISE cutout has non-finite values for %s, skipping", pos)
        return False

    try:
        tmp = img_wise[0].header.copy()
        tmp.update(d_wise.wcs.to_header())
        fits.writeto(wise_outpath, d_wise.data, tmp, overwrite=True)

        first = fits.open(radio_outpath)[0]
        wise = fits.open(wise_outpath)[0]
        arr, _ = reproject_interp(
            wise, WCS(first.header).celestial, shape_out=first.data.shape
        )

        tmp2 = wise.header.copy()
        tmp2.update(WCS(first.header).celestial.to_header())
        repo_path = wise_outpath.replace(".fits", "_repo.fits")
        fits.writeto(repo_path, arr, tmp2, overwrite=True)
        return True

    except Exception as exc:
        logger.warning("WISE reproject failed for %s: %s", pos, exc)
        return False


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def download_cutouts(
    sbid: str,
    emu_tile_path: str,
    catalog_path: str,
    wise_tile_path: str,
    cutout_size: int = 240,
    angular_size_arcmin: float = 8.0,
) -> None:
    """
    Download all EMU + WISE cutouts for sources listed in *catalog_path*.

    Parameters
    ----------
    sbid              : SB identifier string (e.g. "50180")
    emu_tile_path     : Absolute/relative path to the EMU FITS tile
    catalog_path      : Path to the selavy CSV catalog for this tile
    wise_tile_path    : Path to the WISE mosaic FITS for this tile
    cutout_size       : Pixel dimensions of each square cutout
    angular_size_arcmin: Angular size of cutout in arcminutes
    """
    output_fits_path = f"./all_SBs/{sbid}"
    out_folder = f"{output_fits_path}/All_Images_8arcmV1"
    wise_out_folder = f"{output_fits_path}/All_Images_8arcmV1_wise"

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(wise_out_folder, exist_ok=True)

    # -- Open EMU tile -------------------------------------------------------
    logger.info("Opening EMU tile: %s", emu_tile_path)
    img = fits.open(emu_tile_path, relax=True)
    wcs = WCS(img[0].header)

    # -- Open WISE mosaic ----------------------------------------------------
    logger.info("Opening WISE mosaic: %s", wise_tile_path)
    img_wise = fits.open(wise_tile_path, relax=True)
    wcs_wise = WCS(img_wise[0].header)

    # -- Load catalog --------------------------------------------------------
    logger.info("Reading catalog: %s", catalog_path)
    tab = pd.read_csv(catalog_path, header=0)
    tab.rename(
        columns={"ra_deg_cont": "ra_deg_radio", "dec_deg_cont": "dec_deg_radio"},
        inplace=True,
    )
    logger.info("Catalog has %d sources", len(tab))

    angular = angular_size_arcmin * u.arcmin
    img_size = (cutout_size, cutout_size)

    n_radio_ok = 0
    n_wise_ok = 0
    n_radio_fail = 0
    n_wise_fail = 0
    cutout_failed_ids = []

    for _, row in tqdm(tab.iterrows(), total=len(tab), desc=f"SB{sbid} cutouts"):
        ra = row["ra_deg_radio"]
        dec = row["dec_deg_radio"]
        src_id = f"{ra}_{dec}"
        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        # ---- EMU radio cutout ----------------------------------------------
        radio_outpath = f"{out_folder}/{src_id}_emu.fits"
        if not os.path.exists(radio_outpath):
            try:
                d = Cutout2D(
                    img[0].data.squeeze(),
                    position=pos,
                    size=img_size,
                    fill_value=np.nan,
                    wcs=wcs.celestial,
                )
                d.data[~np.isfinite(d.data)] = 0.0
                tmp = img[0].header.copy()
                tmp.update(d.wcs.to_header())
                fits.writeto(radio_outpath, d.data, tmp, overwrite=True)
                n_radio_ok += 1
            except Exception as exc:
                logger.debug("Radio cutout failed for %s: %s", src_id, exc)
                cutout_failed_ids.append(src_id)
                n_radio_fail += 1
                continue  # no point trying WISE without radio

        # ---- WISE IR cutout ------------------------------------------------
        wise_outpath = f"{wise_out_folder}/{src_id}_wise.fits"
        wise_repo_path = wise_outpath.replace(".fits", "_repo.fits")
        if not os.path.exists(wise_outpath) and not os.path.exists(wise_repo_path):
            success = make_wise_cutout(
                pos=pos,
                angular_size=angular,
                cutout_size=cutout_size,
                img_wise=img_wise,
                wcs_wise=wcs_wise,
                wise_outpath=wise_outpath,
                radio_outpath=radio_outpath,
            )
            if success:
                n_wise_ok += 1
            else:
                n_wise_fail += 1

    img.close()
    img_wise.close()

    logger.info(
        "SB%s download complete — radio: %d ok / %d failed | WISE: %d ok / %d failed",
        sbid,
        n_radio_ok,
        n_radio_fail,
        n_wise_ok,
        n_wise_fail,
    )
    if cutout_failed_ids:
        logger.warning(
            "%d radio cutouts failed. First few: %s",
            len(cutout_failed_ids),
            cutout_failed_ids[:5],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    tile_filename = args.Image_SBid
    emu_tile_path = os.path.join(args.images_base_dir, tile_filename)

    # Extract SBid from filename
    sbmatch = re.search(r"SB(\d+)", tile_filename)
    if not sbmatch:
        logger.error(
            "Could not extract SBid from filename '%s'. "
            "Expected pattern *SB#####*.fits",
            tile_filename,
        )
        sys.exit(1)
    sbid = sbmatch.group(1)
    logger.info("Working on SBid: %s", sbid)

    # Validate EMU tile exists
    if not os.path.exists(emu_tile_path):
        logger.error("EMU tile not found: %s", emu_tile_path)
        sys.exit(1)

    # Locate catalog
    catalogs_dir = os.path.dirname(emu_tile_path).replace("images", "catalogs")
    try:
        catalog_path = find_catalog(catalogs_dir, sbid)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Found catalog: %s", catalog_path)

    # Locate WISE mosaic
    wise_tile_path = emu_tile_path.replace(".fits", "_wise_mosaic.fits").replace(
        "/images/", "/images_wise/"
    )
    if not os.path.exists(wise_tile_path):
        logger.error("WISE mosaic not found: %s", wise_tile_path)
        sys.exit(1)
    logger.info("Found WISE mosaic: %s", wise_tile_path)

    download_cutouts(
        sbid=sbid,
        emu_tile_path=emu_tile_path,
        catalog_path=catalog_path,
        wise_tile_path=wise_tile_path,
        cutout_size=args.cutout_size,
        angular_size_arcmin=args.angular_size,
    )


if __name__ == "__main__":
    main()
