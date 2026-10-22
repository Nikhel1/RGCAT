#!/usr/bin/env python
"""
Download_allSources.py
----------------------
For a given EMU tile (identified by its filename), downloads:
  - EMU radio cutouts  →  all_SBs/<SBid>/All_Images_8arcmV1/
  - WISE IR cutouts    →  all_SBs/<SBid>/All_Images_8arcmV1_wise/

Optionally parallelises per-source cutout extraction across CPU cores
using ``concurrent.futures`` from the Python standard library — no extra
dependencies required, works on both HPC (SLURM) and cloud instances.

Usage:
    python Download_allSources.py --Image_SBid image.i.EMU_0044-46.SB50180....fits
    python Download_allSources.py --Image_SBid ... --n_workers 8 --parallel_backend process
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
        "--emu_images_dir",
        type=str,
        default="data/images",
        help="Directory containing EMU FITS tiles (default: data/images)",
    )
    parser.add_argument(
        "--wise_images_dir",
        type=str,
        default="data/images_wise",
        help="Directory containing WISE mosaic FITS files (default: data/images_wise)",
    )
    parser.add_argument(
        "--emu_catalogs_dir",
        type=str,
        default="data/catalogs",
        help="Directory containing selavy CSV catalogs (default: data/catalogs)",
    )
    parser.add_argument(
        "--all_sbs_dir",
        type=str,
        default="data/all_SBs",
        help="Working area for per-SBid cutouts (default: data/all_SBs)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers (default: auto — uses all available "
            "CPU cores). Set to 1 for sequential execution."
        ),
    )
    parser.add_argument(
        "--parallel_backend",
        type=str,
        default="process",
        choices=["process", "thread"],
        help=(
            "Parallel backend: 'process' (ProcessPoolExecutor, default) for "
            "CPU-bound work or 'thread' (ThreadPoolExecutor) when I/O dominates "
            "and the GIL is released by underlying libraries."
        ),
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
# WISE cutout (I/O-optimised — no intermediate disk write / re-read)
# ---------------------------------------------------------------------------
def make_wise_cutout(
    pos: SkyCoord,
    angular_size: u.Quantity,
    cutout_size: int,
    img_wise_data: np.ndarray,
    wcs_wise: WCS,
    radio_wcs: WCS,
    radio_shape: tuple,
    wise_outpath: str,
) -> bool:
    """
    Extract a WISE IR cutout centred on *pos*, reproject to match the radio
    cutout WCS, and write the reprojected result to *wise_outpath*.

    Unlike the original implementation this avoids writing an intermediate
    WISE FITS file and re-opening both the radio and WISE files from disk —
    the WCS and data needed for reprojection are passed in-memory.

    Returns True on success, False on any failure.
    """
    try:
        d_wise = Cutout2D(
            img_wise_data,
            position=pos,
            size=(cutout_size + 30, cutout_size + 30),
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
        # Reproject WISE cutout onto the radio cutout WCS — all in-memory
        arr, _ = reproject_interp(
            (d_wise.data, d_wise.wcs), radio_wcs, shape_out=radio_shape,
        )

        out_header = radio_wcs.to_header()
        repo_path = wise_outpath.replace(".fits", "_repo.fits")
        fits.writeto(repo_path, arr, out_header, overwrite=True)
        return True

    except Exception as exc:
        logger.warning("WISE reproject failed for %s: %s", pos, exc)
        return False


# ---------------------------------------------------------------------------
# Parallel worker state
# ---------------------------------------------------------------------------
# When using ProcessPoolExecutor each worker process needs its own copy of
# the large FITS tile data.  We load them once per process and stash in this
# dict so subsequent _process_source calls reuse the open HDU lists.
# For ThreadPoolExecutor the main thread populates this dict; threads share
# the same memory space.
_worker_state: dict = {}


def _init_worker(emu_tile_path: str, wise_tile_path: str) -> None:
    """Initialiser for ProcessPoolExecutor — opens FITS files once per worker."""
    _worker_state["img"] = fits.open(emu_tile_path, relax=True)
    _worker_state["wcs"] = WCS(_worker_state["img"][0].header)
    _worker_state["img_wise"] = fits.open(wise_tile_path, relax=True)
    _worker_state["wcs_wise"] = WCS(_worker_state["img_wise"][0].header)


def _process_source(
    src_id: str,
    ra: float,
    dec: float,
    out_folder: str,
    wise_out_folder: str,
    cutout_size: int,
    angular: u.Quantity,
) -> dict:
    """
    Process a single source: extract radio cutout and (if needed) WISE cutout.

    Relies on ``_worker_state`` being populated (either by ``_init_worker``
    in a ProcessPoolExecutor, or by the caller for sequential / thread mode).

    Returns a dict with keys: src_id, radio_ok, wise_ok, radio_fail, wise_fail.
    """
    result = {
        "src_id": src_id,
        "radio_ok": False,
        "wise_ok": False,
        "radio_fail": False,
        "wise_fail": False,
    }

    pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    img = _worker_state["img"]
    wcs = _worker_state["wcs"]
    img_wise = _worker_state["img_wise"]
    wcs_wise = _worker_state["wcs_wise"]

    img_size = (cutout_size, cutout_size)

    # ---- EMU radio cutout --------------------------------------------------
    radio_outpath = os.path.join(out_folder, f"{src_id}_emu.fits")
    radio_cutout_wcs = None
    radio_cutout_data = None

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
            result["radio_ok"] = True
            radio_cutout_wcs = d.wcs
            radio_cutout_data = d.data
        except Exception as exc:
            logger.debug("Radio cutout failed for %s: %s", src_id, exc)
            result["radio_fail"] = True
            return result
    else:
        # File already exists — load WCS + data for WISE reprojection target
        try:
            with fits.open(radio_outpath) as existing:
                radio_cutout_wcs = WCS(existing[0].header).celestial
                radio_cutout_data = existing[0].data.squeeze()
            result["radio_ok"] = True
        except Exception:
            return result

    # ---- WISE IR cutout ----------------------------------------------------
    wise_outpath = os.path.join(wise_out_folder, f"{src_id}_wise.fits")
    wise_repo_path = wise_outpath.replace(".fits", "_repo.fits")
    if not os.path.exists(wise_outpath) and not os.path.exists(wise_repo_path):
        success = make_wise_cutout(
            pos=pos,
            angular_size=angular,
            cutout_size=cutout_size,
            img_wise_data=img_wise[0].data.squeeze(),
            wcs_wise=wcs_wise,
            radio_wcs=radio_cutout_wcs,
            radio_shape=radio_cutout_data.shape,
            wise_outpath=wise_outpath,
        )
        if success:
            result["wise_ok"] = True
        else:
            result["wise_fail"] = True

    return result


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
def _accumulate(result: dict, counters: dict, failed_ids: list) -> None:
    """Update mutable *counters* dict and *failed_ids* list from a result."""
    if result["radio_ok"]:
        counters["radio_ok"] += 1
    if result["radio_fail"]:
        counters["radio_fail"] += 1
        failed_ids.append(result["src_id"])
    if result["wise_ok"]:
        counters["wise_ok"] += 1
    if result["wise_fail"]:
        counters["wise_fail"] += 1


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def download_cutouts(
    sbid: str,
    emu_tile_path: str,
    catalog_path: str,
    wise_tile_path: str,
    all_sbs_dir: str = "data/all_SBs",
    cutout_size: int = 240,
    angular_size_arcmin: float = 8.0,
    n_workers: int | None = None,
    parallel_backend: str = "process",
) -> None:
    """
    Download all EMU + WISE cutouts for sources listed in *catalog_path*.

    Parameters
    ----------
    sbid              : SB identifier string (e.g. "50180")
    emu_tile_path     : Absolute/relative path to the EMU FITS tile
    catalog_path      : Path to the selavy CSV catalog for this tile
    wise_tile_path    : Path to the WISE mosaic FITS for this tile
    all_sbs_dir       : Working area for per-SBid cutouts
    cutout_size       : Pixel dimensions of each square cutout
    angular_size_arcmin: Angular size of cutout in arcminutes
    n_workers         : Number of parallel workers (None → auto)
    parallel_backend  : 'process' or 'thread'
    """
    output_fits_path = os.path.join(all_sbs_dir, sbid)
    out_folder = os.path.join(output_fits_path, "All_Images_8arcmV1")
    wise_out_folder = os.path.join(output_fits_path, "All_Images_8arcmV1_wise")

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(wise_out_folder, exist_ok=True)

    # -- Load catalog --------------------------------------------------------
    logger.info("Reading catalog: %s", catalog_path)
    tab = pd.read_csv(catalog_path, header=0)
    tab.rename(
        columns={"ra_deg_cont": "ra_deg_radio", "dec_deg_cont": "dec_deg_radio"},
        inplace=True,
    )
    logger.info("Catalog has %d sources", len(tab))

    angular = angular_size_arcmin * u.arcmin

    # Resolve worker count
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, n_workers)
    logger.info("Using %d worker(s) with '%s' backend", n_workers, parallel_backend)

    # Build the list of source tuples
    sources = [
        (f"{row['ra_deg_radio']}_{row['dec_deg_radio']}",
         row["ra_deg_radio"], row["dec_deg_radio"])
        for _, row in tab.iterrows()
    ]

    counters = {"radio_ok": 0, "wise_ok": 0, "radio_fail": 0, "wise_fail": 0}
    failed_ids: list = []

    # -- Sequential fallback (n_workers == 1) -------------------------------
    if n_workers == 1:
        logger.info("Opening EMU tile: %s", emu_tile_path)
        logger.info("Opening WISE mosaic: %s", wise_tile_path)
        _init_worker(emu_tile_path, wise_tile_path)
        try:
            for src_id, ra, dec in tqdm(
                sources, total=len(sources), desc=f"SB{sbid} cutouts",
            ):
                result = _process_source(
                    src_id=src_id, ra=ra, dec=dec,
                    out_folder=out_folder, wise_out_folder=wise_out_folder,
                    cutout_size=cutout_size, angular=angular,
                )
                _accumulate(result, counters, failed_ids)
        finally:
            _worker_state["img"].close()
            _worker_state["img_wise"].close()
            _worker_state.clear()

    # -- Parallel: ProcessPoolExecutor --------------------------------------
    elif parallel_backend == "process":
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(emu_tile_path, wise_tile_path),
        ) as executor:
            futures = {
                executor.submit(
                    _process_source,
                    src_id=src_id, ra=ra, dec=dec,
                    out_folder=out_folder, wise_out_folder=wise_out_folder,
                    cutout_size=cutout_size, angular=angular,
                ): src_id
                for src_id, ra, dec in sources
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"SB{sbid} cutouts",
            ):
                _accumulate(future.result(), counters, failed_ids)

    # -- Parallel: ThreadPoolExecutor ---------------------------------------
    else:
        logger.info("Opening EMU tile: %s", emu_tile_path)
        logger.info("Opening WISE mosaic: %s", wise_tile_path)
        _init_worker(emu_tile_path, wise_tile_path)
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        _process_source,
                        src_id=src_id, ra=ra, dec=dec,
                        out_folder=out_folder, wise_out_folder=wise_out_folder,
                        cutout_size=cutout_size, angular=angular,
                    ): src_id
                    for src_id, ra, dec in sources
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"SB{sbid} cutouts",
                ):
                    _accumulate(future.result(), counters, failed_ids)
        finally:
            _worker_state["img"].close()
            _worker_state["img_wise"].close()
            _worker_state.clear()

    logger.info(
        "SB%s download complete — radio: %d ok / %d failed | WISE: %d ok / %d failed",
        sbid, counters["radio_ok"], counters["radio_fail"],
        counters["wise_ok"], counters["wise_fail"],
    )
    if failed_ids:
        logger.warning(
            "%d radio cutouts failed. First few: %s",
            len(failed_ids), failed_ids[:5],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    tile_filename = args.Image_SBid
    emu_tile_path = os.path.join(args.emu_images_dir, tile_filename)

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

    # Locate catalog using explicit emu_catalogs_dir
    try:
        catalog_path = find_catalog(args.emu_catalogs_dir, sbid)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Found catalog: %s", catalog_path)

    # Locate WISE mosaic using explicit wise_images_dir
    wise_basename = tile_filename.replace(".fits", "_wise_mosaic.fits")
    wise_tile_path = os.path.join(args.wise_images_dir, wise_basename)
    if not os.path.exists(wise_tile_path):
        logger.error("WISE mosaic not found: %s", wise_tile_path)
        sys.exit(1)
    logger.info("Found WISE mosaic: %s", wise_tile_path)

    download_cutouts(
        sbid=sbid,
        emu_tile_path=emu_tile_path,
        catalog_path=catalog_path,
        wise_tile_path=wise_tile_path,
        all_sbs_dir=args.all_sbs_dir,
        cutout_size=args.cutout_size,
        angular_size_arcmin=args.angular_size,
        n_workers=args.n_workers,
        parallel_backend=args.parallel_backend,
    )


if __name__ == "__main__":
    main()
