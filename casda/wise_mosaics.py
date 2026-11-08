#!/usr/bin/env python
# coding: utf-8
"""
Generate unWISE W1 mosaics for every EMU FITS tile in data/images/.

For each tile the script:
  1. Reads the WCS footprint and pads it by ``wise.footprint_pad_deg`` degrees.
  2. Queries the unWISE tile index to find overlapping coadd IDs.
  3. Downloads the W1 cutout FITS files into a per-tile scratch directory.
  4. Runs the Montage mosaicking pipeline (reproject → background-model → coadd).
  5. Saves the mosaic to ``data/images_wise/<stem>_wise_mosaic.fits``.
  6. Cleans up all scratch files, keeping only the final mosaic.

Tiles listed under ``skip.bad_tiles`` in pipeline.yaml are skipped automatically.

Usage (from repo root):
    python casda/wise_mosaics.py [--config config/pipeline.yaml]
"""

import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import wcs as awcs
from MontagePy.main import (
    mAdd,
    mBgExec,
    mBgModel,
    mDiffFitExec,
    mImgtbl,
    mMakeHdr,
    mOverlaps,
    mProject,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict) -> dict:
    """Return absolute path strings from the paths section."""
    root = Path(__file__).resolve().parent.parent  # repo root
    p = cfg.get("paths", {})
    return {
        "emu_images_dir": root / p.get("emu_images_dir", "data/images"),
        "wise_images_dir": root / p.get("wise_images_dir", "data/images_wise"),
    }


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, streaming.  Returns *dest* on success."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest


def download_wise_tiles(coadd_ids, base_url: str, outdir: Path) -> None:
    """Download W1 unWISE coadd images for each coadd_id into *outdir*."""
    for coadd_id in coadd_ids:
        dest = outdir / f"unwise-{coadd_id}-w1-img-m.fits"
        if dest.exists():
            continue
        dir1 = coadd_id[:3]
        url = f"{base_url}/{dir1}/{coadd_id}/unwise-{coadd_id}-w1-img-m.fits"
        download_file(url, dest)


# ---------------------------------------------------------------------------
# Footprint / tile-index helpers
# ---------------------------------------------------------------------------

def footprint_corners(fits_path: str, pad_deg: float):
    """
    Return four padded sky-coordinate corners of the EMU FITS tile.

    *pad_deg* is added outward in both RA (scaled by 1/cos(dec)) and Dec.
    """
    header = fits.getheader(fits_path)
    w = awcs.WCS(header, naxis=2)
    corners = w.calc_footprint(header)          # (4, 2) array, degrees
    cosdec = np.cos(np.radians(w.wcs.crval[1]))
    offsets = np.array([
        [+pad_deg / cosdec, -pad_deg],
        [+pad_deg / cosdec, +pad_deg],
        [-pad_deg / cosdec, +pad_deg],
        [-pad_deg / cosdec, -pad_deg],
    ])
    corners = corners + offsets
    return [SkyCoord(ra=c[0], dec=c[1], unit="deg", frame="fk5") for c in corners]


def query_tile_index(tile_table_path: str, corners) -> list:
    """
    Return a list of coadd_id strings from the unWISE tile index that overlap
    the bounding box of *corners*.

    Handles the RA wrap-around case (bounding box width ≥ 50°).
    """
    from astropy.io import fits as afits
    from astropy.table import Table

    ra_vals  = [c.ra.deg  for c in corners]
    dec_vals = [c.dec.deg for c in corners]
    ra_min,  ra_max  = min(ra_vals),  max(ra_vals)
    dec_min, dec_max = min(dec_vals), max(dec_vals)

    with afits.open(tile_table_path) as hdul:
        data = Table(hdul[1].data)

    if (ra_max - ra_min) >= 50:
        # Wrap-around: select tiles outside the gap
        mask = ((data["ra"] >= ra_max) | (data["ra"] <= ra_min)) & \
               (data["dec"] >= dec_min) & (data["dec"] <= dec_max)
    else:
        mask = (data["ra"] >= ra_min) & (data["ra"] <= ra_max) & \
               (data["dec"] >= dec_min) & (data["dec"] <= dec_max)

    return list(data["coadd_id"][mask])


# ---------------------------------------------------------------------------
# Montage mosaicking pipeline
# ---------------------------------------------------------------------------

def make_mosaic(outdir: Path, mosaic_path: Path) -> None:
    """
    Run the full Montage pipeline on the FITS files in *outdir* and write the
    final mosaic to *mosaic_path*.
    """
    project_dir = outdir / "project"
    corr_dir    = project_dir / "corrected"
    diff_dir    = project_dir / "diff"
    for d in (project_dir, corr_dir, diff_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_table        = str(outdir / "rimages.tbl")
    template_hdr     = str(outdir / "template.hdr")
    reprojected_tbl  = str(project_dir / "pimages.tbl")
    overlaps_tbl     = str(project_dir / "diff.tbl")
    fits_tbl         = str(project_dir / "fits.tbl")
    bg_model_tbl     = str(corr_dir / "corrections.tbl")
    cimg_tbl         = str(corr_dir / "cimages.tbl")

    # Build image tables and header template
    mImgtbl(str(outdir), raw_table)
    mMakeHdr(raw_table, template_hdr)

    # Reproject each image onto the common grid
    print("  Reprojecting images...")
    fits_files = [f for f in os.listdir(outdir) if f.endswith(".fits")]
    for fname in fits_files:
        mProject(str(outdir / fname), str(project_dir / fname), template_hdr)

    mImgtbl(str(project_dir), reprojected_tbl)

    # Background modelling
    mOverlaps(reprojected_tbl, overlaps_tbl)
    mDiffFitExec(str(project_dir), overlaps_tbl, template_hdr, str(diff_dir), fits_tbl)
    print("  Modelling background...")
    mBgModel(reprojected_tbl, fits_tbl, bg_model_tbl)
    print("  Applying background corrections...")
    mBgExec(str(project_dir), reprojected_tbl, bg_model_tbl, str(corr_dir))
    mImgtbl(str(corr_dir), cimg_tbl)

    # Final coadd
    print("  Creating final mosaic...")
    mAdd("./", cimg_tbl, template_hdr, str(mosaic_path), haveAreas=True, shrink=True)


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _remove_files_keep_nfs(dir_path: Path) -> None:
    """Delete all regular files under *dir_path*, skipping NFS lock files."""
    for item in dir_path.rglob("*"):
        if item.is_file() and not item.name.startswith(".nfs"):
            try:
                item.unlink()
            except OSError:
                pass


def _kill_nfs_lock_holders(dir_path: Path) -> None:
    """
    Use lsof to find processes holding .nfs lock files under *dir_path* and
    send them SIGTERM so the kernel can release the locks.
    Silently skips if lsof is not available.
    """
    import shutil as _shutil
    import signal
    if not _shutil.which("lsof"):
        return
    try:
        import subprocess as _sp
        result = _sp.run(
            ["lsof", "+D", str(dir_path)],
            capture_output=True, text=True, timeout=15,
        )
        pids = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                pids.add(int(parts[1]))
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if pids:
            time.sleep(1)  # give processes a moment to release locks
    except Exception:
        pass


def cleanup_scratch(outdir: Path) -> None:
    """Remove the per-tile scratch directory and the _area sidecar file.

    Strategy:
    1. Delete all regular files except .nfs lock files (these are OS lock
       handles, not copies of data — typically a few bytes each).
    2. Try shutil.rmtree on the directory.
    3. If that fails because .nfs locks remain, use lsof to find and SIGTERM
       the processes holding them, then retry rmtree.
    4. If locks still remain after the retry, leave the (now empty-except-
       for-locks) directory — it uses negligible space and the shell wrapper
       will clean it up before the next attempt.
    """
    area_file = Path(str(outdir).rstrip("/") + "_mosaic_area.fits")

    # Step 1: remove all real files (GBs of unWISE FITS + intermediates)
    _remove_files_keep_nfs(outdir)

    def _onerror(func, path, _exc):
        try:
            os.chmod(path, 0o777)
            func(path)
        except OSError:
            pass

    if outdir.is_dir():
        # Step 2: first rmtree attempt
        try:
            shutil.rmtree(outdir, onerror=_onerror)
        except Exception:
            # Step 3: kill lock holders and retry
            _kill_nfs_lock_holders(outdir)
            try:
                shutil.rmtree(outdir, onerror=_onerror)
            except Exception:
                # Step 4: only .nfs stubs remain — negligible size, leave them
                nfs_count = sum(
                    1 for p in outdir.rglob("*")
                    if p.is_file() and p.name.startswith(".nfs")
                )
                other = [
                    p for p in outdir.rglob("*")
                    if p.is_file() and not p.name.startswith(".nfs")
                ]
                if other:
                    print(f"  Warning: {len(other)} unexpected file(s) remain "
                          f"in {outdir.name} after cleanup.")
                else:
                    # Truly just NFS stubs — safe to leave, log quietly
                    print(f"  Note: {nfs_count} NFS lock stub(s) remain in "
                          f"{outdir.name} — will be cleaned up by the shell "
                          f"wrapper before the next run.")

    if area_file.is_file():
        try:
            area_file.unlink()
        except Exception as e:
            print(f"  Warning: could not remove {area_file.name}: {e}")


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(
    fits_path: str,
    emu_images_dir: Path,
    wise_images_dir: Path,
    tiles_table: str,
    base_url: str,
    pad_deg: float,
    max_tiles: int,
    bad_sbids: set,
) -> bool:
    """
    Generate a WISE mosaic for one EMU tile.

    Returns True on success or skip, False on error.
    """
    stem = Path(fits_path).stem           # e.g. image.i.EMU_0044-46.SB50180...conv
    mosaic_path = wise_images_dir / f"{stem}_wise_mosaic.fits"
    outdir      = wise_images_dir / f"{stem}_wise"

    # Extract SBID to check against bad_tiles list
    import re
    m = re.search(r"SB(\d+)", stem)
    sbid = m.group(1) if m else None

    # Skip already-done or bad tiles
    if mosaic_path.exists():
        print(f"  Mosaic exists, skipping: {mosaic_path.name}")
        return True
    if sbid and sbid in bad_sbids:
        print(f"  Skipping bad tile SBID {sbid} ({Path(fits_path).name})")
        return True

    print(f"\nProcessing: {Path(fits_path).name}")

    # Find overlapping unWISE tiles
    try:
        corners = footprint_corners(fits_path, pad_deg)
        coadd_ids = query_tile_index(tiles_table, corners)
    except Exception as e:
        print(f"  ERROR computing footprint: {e}")
        return False

    print(f"  Found {len(coadd_ids)} unWISE tile(s).")
    if len(coadd_ids) > max_tiles:
        print(f"  Too many tiles ({len(coadd_ids)} > {max_tiles}), "
              f"likely a WCS problem — skipping.")
        return True   # not a retriable error

    if len(coadd_ids) == 0:
        print("  No unWISE tiles found for this footprint, skipping.")
        return True

    outdir.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"  Downloading {len(coadd_ids)} unWISE image(s)...")
    try:
        download_wise_tiles(coadd_ids, base_url, outdir)
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        cleanup_scratch(outdir)
        return False

    # Mosaic
    try:
        make_mosaic(outdir, mosaic_path)
    except Exception as e:
        print(f"  ERROR mosaicking: {e}")
        cleanup_scratch(outdir)
        return False

    # Verify output
    if not mosaic_path.exists():
        print(f"  ERROR: mosaic not created at {mosaic_path}")
        cleanup_scratch(outdir)
        return False

    try:
        with fits.open(mosaic_path) as hdul:
            shape = hdul[0].data.shape
            print(f"  Mosaic OK — shape: {shape}")
    except Exception as e:
        print(f"  WARNING: mosaic created but could not be opened: {e}")

    cleanup_scratch(outdir)
    print(f"  → {mosaic_path.name}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate unWISE W1 mosaics for EMU FITS tiles."
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    wise_cfg = cfg.get("wise", {})
    skip_cfg = cfg.get("skip", {})

    emu_images_dir  = paths["emu_images_dir"]
    wise_images_dir = paths["wise_images_dir"]
    wise_images_dir.mkdir(parents=True, exist_ok=True)

    tiles_url  = wise_cfg.get("tiles_url",
                              "http://unwise.me/data/allwise/unwise-coadds/fulldepth/tiles.fits")
    base_url   = wise_cfg.get("images_base_url",
                              "http://unwise.me/data/neo3/unwise-coadds/fulldepth")
    max_tiles  = int(wise_cfg.get("max_tiles_per_footprint", 75))
    pad_deg    = float(wise_cfg.get("footprint_pad_deg", 1.5))
    bad_sbids  = set(str(s) for s in skip_cfg.get("bad_tiles", []))

    # Download the unWISE tile index once
    tiles_table = wise_images_dir / "tiles.fits"
    if not tiles_table.exists():
        print(f"Downloading unWISE tile index from {tiles_url} ...")
        download_file(tiles_url, tiles_table)
        print("  Done.")
    else:
        print(f"Using cached unWISE tile index: {tiles_table}")

    # Collect EMU tiles
    emu_fits = sorted(glob.glob(str(emu_images_dir / "*.fits")))
    if not emu_fits:
        print(f"No FITS files found in {emu_images_dir}")
        sys.exit(0)
    print(f"Found {len(emu_fits)} EMU tile(s) in {emu_images_dir}")

    failed = []
    for fits_path in emu_fits:
        ok = process_tile(
            fits_path=fits_path,
            emu_images_dir=emu_images_dir,
            wise_images_dir=wise_images_dir,
            tiles_table=str(tiles_table),
            base_url=base_url,
            pad_deg=pad_deg,
            max_tiles=max_tiles,
            bad_sbids=bad_sbids,
        )
        if not ok:
            failed.append(Path(fits_path).name)

    # Final summary
    if failed:
        print(f"\n{len(failed)} tile(s) failed (will be retried on next run):")
        for fn in failed:
            print(f"  {fn}")
        sys.exit(1)
    else:
        print("\nAll tiles processed successfully.")


if __name__ == "__main__":
    main()
