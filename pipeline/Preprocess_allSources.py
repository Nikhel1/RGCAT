#!/usr/bin/env python
"""
Preprocess_allSources.py
------------------------
For a given EMU tile (identified by its SBid filename), preprocesses the
previously downloaded radio + WISE cutouts and writes:
  - PNG RGB composites  →  all_SBs/<SBid>/All2017/
  - Preprocessed FITS   →  all_SBs/<SBid>/All_fits/

Optionally parallelises per-source preprocessing across CPU cores
using ``concurrent.futures`` from the Python standard library — no extra
dependencies required, works on both HPC (SLURM) and cloud instances.

Usage:
    python Preprocess_allSources.py --Image_SBid image.i.EMU_0044-46.SB50180....fits
    python Preprocess_allSources.py --Image_SBid ... --n_workers 8 --parallel_backend process
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

import astropink as pu
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from PIL import Image
from reproject import reproject_interp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=AstropyWarning)
np.seterr(all="ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Preprocess_allSources")

# Suppress verbose astropink binary wrapper logging
logging.getLogger("astropink.binwrap").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EMU + WISE cutouts for all sources in a tile."
    )
    parser.add_argument(
        "--Image_SBid",
        type=str,
        required=True,
        help="Filename of the EMU FITS tile (basename only)",
    )
    parser.add_argument(
        "--emu_images_dir",
        type=str,
        default="data/images",
        help="Directory containing EMU FITS tiles (default: data/images)",
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
        "--cutout_size",
        type=int,
        default=240,
        help="Pixel size of each cutout (default: 240)",
    )
    parser.add_argument(
        "--epmask",
        type=int,
        default=5,
        help="Minimum island pixel count for EMU segmentation (default: 5)",
    )
    parser.add_argument(
        "--wpmask",
        type=int,
        default=5,
        help="Minimum island pixel count for WISE segmentation (default: 5)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Detection threshold in sigma for island segmentation (default: 3.0)",
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
            "CPU-bound work or 'thread' (ThreadPoolExecutor) when I/O dominates."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def find_catalog(catalogs_dir: str, sbid: str) -> str:
    pattern = os.path.join(catalogs_dir, f"*{sbid}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No catalog CSV found matching: {pattern}")
    if len(matches) > 1:
        logger.warning("Multiple catalogs match SBid %s — using: %s", sbid, matches[0])
    return matches[0]


# ---------------------------------------------------------------------------
# RMS estimation
# ---------------------------------------------------------------------------
def rms_estimate(
    data: np.ndarray,
    mode: str = "mad",
    clip_rounds: int = 2,
    outlier_thres: float = 3.0,
) -> float:
    """Estimate RMS via iterative sigma-clipping using std or MAD."""
    data = data[np.isfinite(data)]

    if mode == "mad":
        clipping_func = lambda d: np.median(np.abs(d - np.median(d)))
    elif mode == "std":
        clipping_func = lambda d: np.std(d)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'mad' or 'std'.")

    for _ in range(clip_rounds):
        data = data[np.abs(data) < outlier_thres * clipping_func(data)]

    return float(np.std(data))


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def radio_preprocess(
    data: np.ndarray,
    pmask: int = 2,
    lower: float = 3.0,
    outer_pixels: bool = True,
) -> np.ndarray:
    """Log-scale normalise radio emission islands above *lower* × RMS."""
    nx, ny = data.shape
    noise = rms_estimate(data.flatten(), mode="mad")
    if np.isnan(noise):
        noise = rms_estimate(data.flatten(), mode="std")

    img_scale = np.zeros_like(data)
    for mask in pu.island_segmentation(data, lower * noise):
        if np.sum(mask) <= pmask:
            continue
        argmask = np.argwhere(mask)
        if any(argmask[:, 0] == ny / 2) and any(argmask[:, 1] == nx / 2):
            img_scale[mask] = pu.minmax(np.log10(data[mask]))
        if outer_pixels:
            img_scale[mask] = pu.minmax(np.log10(data[mask]))
        img_scale[~np.isfinite(img_scale)] = 0.0

    return img_scale.astype(np.float32)


def wise_preprocess(data: np.ndarray, wpmask: int = 2) -> np.ndarray:
    """Log-scale normalise WISE IR emission islands above median + 3×MAD."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    noise = med + 3 * mad

    img_scale = np.zeros_like(data)
    for mask in pu.island_segmentation(data, noise):
        if np.sum(mask) <= wpmask:
            continue
        img_scale[mask] = pu.minmax(np.log10(data[mask]))

    return img_scale.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-source preprocessing (pure function — reads input, returns arrays)
# ---------------------------------------------------------------------------
def preprocess_source(
    src_id: str,
    ra: float,
    dec: float,
    cutout_size: int,
    inpath_radio: str,
    inpath_wise: str,
    epmask: int = 5,
    wpmask: int = 5,
    lower: float = 3.0,
    outer_pixels: bool = True,
) -> tuple | None:
    """
    Load, reproject, and preprocess a single source's radio + WISE cutouts.

    Returns (src_id, e_proc, e_raw, w_proc, reproject_wcs) or None on failure.
    """
    emu_src = os.path.join(inpath_radio, f"{src_id}_emu.fits")

    try:
        dlist = fits.open(emu_src)
        d = dlist[0]
    except Exception as exc:
        logger.warning("Cannot open radio FITS for %s: %s", src_id, exc)
        return None

    d_wcs = WCS(d.header).celestial
    d.data = np.squeeze(d.data)
    d.header = d_wcs.to_header()

    # Build centred reprojection WCS
    reproject_wcs_header = d.header.copy()
    reproject_wcs_header["CRVAL1"] = ra
    reproject_wcs_header["CRVAL2"] = dec
    reproject_wcs_header["CRPIX1"] = cutout_size // 2
    reproject_wcs_header["CRPIX2"] = cutout_size // 2
    reproject_wcs = WCS(reproject_wcs_header).celestial

    try:
        e_new_data, _ = reproject_interp(
            d, reproject_wcs, shape_out=(cutout_size, cutout_size),
        )
    except Exception as exc:
        logger.warning("Reproject failed for %s: %s", src_id, exc)
        dlist.close()
        return None
    dlist.close()

    # Raw radio data (for FITS output)
    e_raw = e_new_data.astype(np.float32)

    # Preprocessed radio data
    e_proc = radio_preprocess(e_new_data, epmask, lower, outer_pixels)
    e_proc[~np.isfinite(e_proc)] = 0.0
    if np.max(e_proc) <= 0:
        return None

    # ---- WISE --------------------------------------------------------------
    # Download_allSources writes reprojected WISE cutouts as *_wise_repo.fits.
    # The intermediate *_wise.fits only exists if reprojection was not needed.
    wise_src = os.path.join(inpath_wise, f"{src_id}_wise_repo.fits")
    if not os.path.exists(wise_src):
        wise_src = os.path.join(inpath_wise, f"{src_id}_wise.fits")
    try:
        wlist = fits.open(wise_src)
        w = wlist[0]
    except Exception as exc:
        logger.warning("Cannot open WISE FITS for %s: %s", src_id, exc)
        return None

    try:
        w_new_data, _ = reproject_interp(
            w, reproject_wcs,
            shape_out=(cutout_size, cutout_size),
        )
    except Exception as exc:
        logger.warning("WISE reproject failed for %s: %s", src_id, exc)
        wlist.close()
        return None
    wlist.close()

    w_proc = wise_preprocess(w_new_data, wpmask)
    w_proc[~np.isfinite(w_proc)] = 0.0
    if np.max(w_proc) <= 0:
        return None

    return src_id, e_proc, e_raw, w_proc, reproject_wcs


# ---------------------------------------------------------------------------
# Worker: preprocess + write outputs (self-contained, returns status only)
# ---------------------------------------------------------------------------
def _preprocess_and_write(
    src_id: str,
    ra: float,
    dec: float,
    cutout_size: int,
    inpath_radio: str,
    inpath_wise: str,
    outpath_png: str,
    outpath_fits: str,
    epmask: int = 5,
    wpmask: int = 5,
    lower: float = 3.0,
) -> dict:
    """
    Process one source end-to-end: load, reproject, preprocess, write outputs.

    Returns a status dict: {src_id, ok (bool), skipped (bool)}.
    """
    fits_out = os.path.join(outpath_fits, f"{src_id}_emu.fits")
    png_out = os.path.join(outpath_png, f"{src_id}_rgb.png")

    # Skip if outputs already exist
    if os.path.exists(fits_out) and os.path.exists(png_out):
        return {"src_id": src_id, "ok": True, "skipped": True}

    result = preprocess_source(
        src_id=src_id,
        ra=ra,
        dec=dec,
        cutout_size=cutout_size,
        inpath_radio=inpath_radio,
        inpath_wise=inpath_wise,
        epmask=epmask,
        wpmask=wpmask,
        lower=lower,
    )
    if result is None:
        return {"src_id": src_id, "ok": False, "skipped": False}

    _, e_proc, e_raw, w_proc, reproject_wcs = result

    # ---- Build RGB PNG (radio G+B channels, WISE R channel) ---------------
    data_16 = ((2**16 - 1) * e_proc).astype(np.uint16)
    img_rgb = np.zeros((data_16.shape[0], data_16.shape[1], 3), dtype=np.uint8)
    img_rgb[:, :, 2] = data_16 >> 8   # radio → blue (high byte)
    img_rgb[:, :, 1] = data_16 & 0xFF  # radio → green (low byte)

    # WISE → red channel
    w_32 = ((2**16 - 1) * w_proc).astype(np.uint32)
    img_rgb[:, :, 0] = w_32 >> 8

    if not os.path.exists(png_out):
        Image.fromarray(img_rgb, "RGB").save(png_out, "PNG")

    # ---- Save preprocessed radio FITS -------------------------------------
    if not os.path.exists(fits_out):
        header = reproject_wcs.to_header()
        hdu = fits.PrimaryHDU(data=e_raw, header=header)
        hdu.writeto(fits_out, overwrite=True)

    return {"src_id": src_id, "ok": True, "skipped": False}


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def preprocess_tile(
    sbid: str,
    catalog_path: str,
    all_sbs_dir: str = "data/all_SBs",
    cutout_size: int = 240,
    epmask: int = 5,
    wpmask: int = 5,
    sigma: float = 3.0,
    n_workers: int | None = None,
    parallel_backend: str = "process",
) -> None:
    """
    Preprocess all sources for a given SBid and write outputs.
    """
    output_fits_path = os.path.join(all_sbs_dir, sbid)
    inpath_radio = os.path.join(output_fits_path, "All_Images_8arcmV1")
    inpath_wise = os.path.join(output_fits_path, "All_Images_8arcmV1_wise")
    outpath_png = os.path.join(output_fits_path, "All2017")
    outpath_fits = os.path.join(output_fits_path, "All_fits")

    os.makedirs(outpath_png, exist_ok=True)
    os.makedirs(outpath_fits, exist_ok=True)

    logger.info("Reading catalog: %s", catalog_path)
    tab = pd.read_csv(catalog_path, header=0)
    # Support both old and new column naming conventions
    if "ra_deg_cont" not in tab.columns and "ra_deg_radio" in tab.columns:
        tab.rename(
            columns={"ra_deg_radio": "ra_deg_cont", "dec_deg_radio": "dec_deg_cont"},
            inplace=True,
        )
    logger.info("Processing %d sources for SBid %s", len(tab), sbid)

    # Resolve worker count
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, n_workers)
    logger.info("Using %d worker(s) with '%s' backend", n_workers, parallel_backend)

    # Build the list of source tuples
    sources = [
        (f"{row['ra_deg_cont']}_{row['dec_deg_cont']}",
         float(row["ra_deg_cont"]), float(row["dec_deg_cont"]))
        for _, row in tab.iterrows()
    ]

    n_ok = 0
    n_fail = 0

    # -- Sequential (n_workers == 1) ----------------------------------------
    if n_workers == 1:
        for src_id, ra, dec in tqdm(
            sources, total=len(sources), desc=f"SB{sbid} preprocess",
        ):
            result = _preprocess_and_write(
                src_id=src_id, ra=ra, dec=dec,
                cutout_size=cutout_size,
                inpath_radio=inpath_radio, inpath_wise=inpath_wise,
                outpath_png=outpath_png, outpath_fits=outpath_fits,
                epmask=epmask, wpmask=wpmask, lower=sigma,
            )
            if result["ok"]:
                n_ok += 1
            else:
                n_fail += 1

    # -- Parallel: ProcessPoolExecutor --------------------------------------
    elif parallel_backend == "process":
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _preprocess_and_write,
                    src_id=src_id, ra=ra, dec=dec,
                    cutout_size=cutout_size,
                    inpath_radio=inpath_radio, inpath_wise=inpath_wise,
                    outpath_png=outpath_png, outpath_fits=outpath_fits,
                    epmask=epmask, wpmask=wpmask, lower=sigma,
                ): src_id
                for src_id, ra, dec in sources
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"SB{sbid} preprocess",
            ):
                result = future.result()
                if result["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1

    # -- Parallel: ThreadPoolExecutor ---------------------------------------
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _preprocess_and_write,
                    src_id=src_id, ra=ra, dec=dec,
                    cutout_size=cutout_size,
                    inpath_radio=inpath_radio, inpath_wise=inpath_wise,
                    outpath_png=outpath_png, outpath_fits=outpath_fits,
                    epmask=epmask, wpmask=wpmask, lower=sigma,
                ): src_id
                for src_id, ra, dec in sources
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"SB{sbid} preprocess",
            ):
                result = future.result()
                if result["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1

    logger.info(
        "SB%s preprocessing complete — %d ok / %d failed", sbid, n_ok, n_fail
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    tile_filename = args.Image_SBid
    emu_tile_path = os.path.join(args.emu_images_dir, tile_filename)

    sbmatch = re.search(r"SB(\d+)", tile_filename)
    if not sbmatch:
        logger.error(
            "Could not extract SBid from '%s'. Expected *SB#####*.fits", tile_filename
        )
        sys.exit(1)
    sbid = sbmatch.group(1)
    logger.info("Working on SBid: %s", sbid)

    if not os.path.exists(emu_tile_path):
        logger.error("EMU tile not found: %s", emu_tile_path)
        sys.exit(1)

    try:
        catalog_path = find_catalog(args.emu_catalogs_dir, sbid)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Found catalog: %s", catalog_path)

    preprocess_tile(
        sbid=sbid,
        catalog_path=catalog_path,
        all_sbs_dir=args.all_sbs_dir,
        cutout_size=args.cutout_size,
        epmask=args.epmask,
        wpmask=args.wpmask,
        sigma=args.sigma,
        n_workers=args.n_workers,
        parallel_backend=args.parallel_backend,
    )


if __name__ == "__main__":
    main()
