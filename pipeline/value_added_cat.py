#!/usr/bin/env python
"""
value_added_cat.py
------------------
Cross-matches the consolidated Gal-DINO catalogue for a single SBid against:
  - CatWISE2020 (infrared photometry, W1/W2)
  - DESI DR10 Legacy Survey (optical photometry, photometric redshifts)

Reads:
  <output_dir>/<SBid>/cons_catalogs/consolidated_catalog_lastv1.csv
  <wise_catalogs_dir>/<SBid>_catwise_catalog.csv
  <desi_catalogs_dir>/<SBid>_desi_dr10_catalog.csv        (optional)
  <desi_catalogs_dir>/<SBid>_desi_dr10_z_catalog.csv      (optional)

Writes:
  value_added_catwise_ls_all.csv / .fits
  value_added_catwise_ls_allExtended.csv / .fits

Usage:
    python value_added_cat.py --SBid 50180
"""

import argparse
import logging
import os
import sys

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("value_added_cat")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate value-added catalogue by cross-matching with CatWISE and DESI."
    )
    parser.add_argument(
        "--SBid",
        type=str,
        required=True,
        help="Scheduling Block identifier (e.g. 50180)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/CatV1",
        help="Base directory for Gal-DINO outputs (default: outputs/CatV1)",
    )
    parser.add_argument(
        "--wise_catalogs_dir",
        type=str,
        default="data/catalogs_wise",
        help="Directory containing CatWISE CSV files (default: data/catalogs_wise)",
    )
    parser.add_argument(
        "--desi_catalogs_dir",
        type=str,
        default="data/catalogs_desi",
        help="Directory containing DESI DR10 CSV files (default: data/catalogs_desi)",
    )
    parser.add_argument(
        "--cat_version",
        type=str,
        default="v1",
        help="Catalogue version string used in filenames (default: v1)",
    )
    return parser


# ---------------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------------
def _build_output_schema() -> list:
    """Return the ordered list of output column names."""
    return [
        # Gal-DINO columns
        "EMU_source_name", "EMU_ra_deg", "EMU_dec_deg",
        "EMU_flux_peak", "EMU_flux_int", "EMU_img_id",
        "EMU_radio_bbox", "EMU_PA", "EMU_ra_cen_bbox", "EMU_dec_cen_bbox",
        "EMU_type", "EMU_score", "EMU_n_selavy_components",
        # Selavy columns
        "selavy_island_id", "selavy_component_id", "selavy_component_name",
        "selavy_ra_hms_cont", "selavy_dec_hms_cont",
        "selavy_ra_deg_cont", "selavy_dec_deg_cont",
        "selavy_ra_err", "selavy_dec_err",
        "selavy_flux_peak", "selavy_flux_peak_err",
        "selavy_flux_int", "selavy_flux_int_err",
        "selavy_spectral_index",
        # CatWISE columns
        "catwise_source_id", "catwise_source_name",
        "catwise_ra", "catwise_dec",
        "catwise_w1mag", "catwise_w1sigm",
        "catwise_w2mag", "catwise_w2sigm",
        "catwise_separation",
        # DESI Legacy Survey columns
        "ls_id", "ls_ra", "ls_dec", "ls_type",
        "ls_pmra", "ls_pmdec", "ls_parallax",
        "ls_mag_g", "ls_mag_r", "ls_mag_i", "ls_mag_z",
        "ls_mag_w1", "ls_mag_w2", "ls_mag_w3", "ls_mag_w4",
        "ls_z_id", "ls_z_phot_median", "ls_z_phot_median_i",
        "ls_z_phot_std_i", "ls_z_spec",
    ]


def build_tab2(tab: pd.DataFrame) -> pd.DataFrame:
    """Copy and rename columns from the consolidated catalogue into the
    value-added output schema, renaming morphology labels to the public
    convention (C → Pec, S → C, FR-X → FR-x).
    """
    cols = _build_output_schema()
    tab2 = pd.DataFrame(columns=cols)

    tab2["EMU_source_name"]         = tab["filename_cons"]
    tab2["EMU_ra_deg"]              = tab["wise_ra_cons"]
    tab2["EMU_dec_deg"]             = tab["wise_dec_cons"]
    tab2["EMU_flux_peak"]           = tab["flux_peak_cons"]
    tab2["EMU_flux_int"]            = tab["flux_int_cons"]
    tab2["EMU_img_id"]              = tab["img_id_cons"]
    tab2["EMU_radio_bbox"]          = tab["radio_bbox_cons"]
    tab2["EMU_PA"]                  = tab["PA"]
    tab2["EMU_ra_cen_bbox"]         = tab["ra_deg_cons"]
    tab2["EMU_dec_cen_bbox"]        = tab["dec_deg_cons"]
    tab2["EMU_type"]                = tab["type"]
    tab2.replace({"EMU_type": {"C": "Pec", "S": "C", "FR-X": "FR-x"}}, inplace=True)
    tab2["EMU_score"]               = tab["score"]
    tab2["EMU_n_selavy_components"] = tab["n_components"]

    tab2["selavy_island_id"]        = tab["island_id"]
    tab2["selavy_component_id"]     = tab["component_id"]
    tab2["selavy_component_name"]   = tab["component_name"]
    tab2["selavy_ra_hms_cont"]      = tab["ra_hms_cont"]
    tab2["selavy_dec_hms_cont"]     = tab["dec_dms_cont"]
    tab2["selavy_ra_deg_cont"]      = tab["ra_deg_cont"]
    tab2["selavy_dec_deg_cont"]     = tab["dec_deg_cont"]
    tab2["selavy_ra_err"]           = tab["ra_err"]
    tab2["selavy_dec_err"]          = tab["dec_err"]
    tab2["selavy_flux_peak"]        = tab["flux_peak"]
    tab2["selavy_flux_peak_err"]    = tab["flux_peak_err"]
    tab2["selavy_flux_int"]         = tab["flux_int"]
    tab2["selavy_flux_int_err"]     = tab["flux_int_err"]
    tab2["selavy_spectral_index"]   = tab["spectral_index"]

    # CatWISE / DESI columns initialised to None — filled later
    for col in cols:
        if col.startswith(("catwise_", "ls_")) and col not in tab2.columns:
            tab2[col] = None

    return tab2


# ---------------------------------------------------------------------------
# CatWISE cross-match
# ---------------------------------------------------------------------------
def crossmatch_catwise(tab2: pd.DataFrame, tab_wise: pd.DataFrame) -> pd.DataFrame:
    """Cross-match tab2 against CatWISE within 10 arcsec (3 arcsec for
    compact sources).  Updates tab2 in place and returns it.
    """
    coords_emu  = SkyCoord(
        ra=tab2["EMU_ra_deg"].values * u.degree,
        dec=tab2["EMU_dec_deg"].values * u.degree,
        frame="icrs",
    )
    coords_wise = SkyCoord(
        ra=tab_wise["ra"].values * u.degree,
        dec=tab_wise["dec"].values * u.degree,
        frame="icrs",
    )

    idx, d2d, _ = coords_emu.match_to_catalog_sky(coords_wise)
    sep_arcsec  = d2d.arcsecond

    wise_cols = {
        "catwise_source_id":   "source_id",
        "catwise_source_name": "source_name",
        "catwise_ra":          "ra",
        "catwise_dec":         "dec",
        "catwise_w1mag":       "w1mag",
        "catwise_w1sigm":      "w1sigm",
        "catwise_w2mag":       "w2mag",
        "catwise_w2sigm":      "w2sigm",
    }
    for out_col, in_col in wise_cols.items():
        tab2[out_col] = tab_wise[in_col].values[idx]
    tab2["catwise_separation"] = sep_arcsec

    # Mask out matches beyond 10 arcsec
    far = sep_arcsec > 10
    for col in list(wise_cols.keys()) + ["catwise_separation"]:
        tab2.loc[far, col] = np.nan

    # Tighter 3 arcsec cut for compact sources
    compact_far = (sep_arcsec > 3) & (tab2["EMU_type"] == "C")
    for col in list(wise_cols.keys()) + ["catwise_separation"]:
        tab2.loc[compact_far, col] = np.nan

    # Refine host position with CatWISE coords where available
    matched = tab2["catwise_separation"].notna()
    tab2.loc[matched, "EMU_ra_deg"]  = tab2.loc[matched, "catwise_ra"]
    tab2.loc[matched, "EMU_dec_deg"] = tab2.loc[matched, "catwise_dec"]

    logger.info(
        "CatWISE: %d/%d sources matched within 10 arcsec",
        int(matched.sum()), len(tab2),
    )
    return tab2


# ---------------------------------------------------------------------------
# DESI cross-match
# ---------------------------------------------------------------------------
def crossmatch_desi(tab2: pd.DataFrame, tab_desi: pd.DataFrame) -> pd.DataFrame:
    """Cross-match tab2 against DESI DR10 within 5 arcsec.
    Updates tab2 in place and returns it.
    """
    coords_emu  = SkyCoord(
        ra=tab2["EMU_ra_deg"].values * u.degree,
        dec=tab2["EMU_dec_deg"].values * u.degree,
        frame="icrs",
    )
    coords_desi = SkyCoord(
        ra=tab_desi["ra"].values * u.degree,
        dec=tab_desi["dec"].values * u.degree,
        frame="icrs",
    )

    idx, d2d, _ = coords_emu.match_to_catalog_sky(coords_desi)
    sep_arcsec  = d2d.arcsecond

    desi_cols = {
        "ls_id":       "ls_id",
        "ls_ra":       "ra",
        "ls_dec":      "dec",
        "ls_type":     "type",
        "ls_pmra":     "pmra",
        "ls_pmdec":    "pmdec",
        "ls_parallax": "parallax",
        "ls_mag_g":    "mag_g",
        "ls_mag_r":    "mag_r",
        "ls_mag_i":    "mag_i",
        "ls_mag_z":    "mag_z",
        "ls_mag_w1":   "mag_w1",
        "ls_mag_w2":   "mag_w2",
        "ls_mag_w3":   "mag_w3",
        "ls_mag_w4":   "mag_w4",
    }
    for out_col, in_col in desi_cols.items():
        tab2[out_col] = tab_desi[in_col].values[idx]

    far = sep_arcsec > 5
    for col in desi_cols:
        tab2.loc[far, col] = np.nan

    # Tighter cut for compact sources using catwise separation as proxy
    compact_far = (tab2["catwise_separation"] > 1) & (tab2["EMU_type"] == "C")
    for col in desi_cols:
        tab2.loc[compact_far, col] = np.nan

    logger.info(
        "DESI DR10: %d/%d sources matched within 5 arcsec",
        int((sep_arcsec <= 5).sum()), len(tab2),
    )
    return tab2


def add_desi_photoz(
    tab2: pd.DataFrame, tab_desi_z: pd.DataFrame
) -> pd.DataFrame:
    """Merge DESI photo-z table into tab2 on ls_id."""
    merged = tab2.merge(
        tab_desi_z[["ls_id", "z_phot_median", "z_phot_median_i", "z_phot_std_i", "z_spec"]],
        on="ls_id",
        how="left",
    )
    tab2["ls_z_id"]           = merged["ls_id"].where(merged["z_phot_median"].notna())
    tab2["ls_z_phot_median"]  = merged["z_phot_median"]
    tab2["ls_z_phot_median_i"]= merged["z_phot_median_i"]
    tab2["ls_z_phot_std_i"]   = merged["z_phot_std_i"]
    tab2["ls_z_spec"]         = merged["z_spec"]
    logger.info(
        "DESI photo-z: %d sources with redshift",
        int(tab2["ls_z_phot_median"].notna().sum()),
    )
    return tab2


# ---------------------------------------------------------------------------
# IAU name builder
# ---------------------------------------------------------------------------
def assign_iau_names(tab2: pd.DataFrame) -> pd.DataFrame:
    """Replace EMU_source_name with IAU-style 'EMU JHHMMSS±DDMMSS' names."""
    coords = SkyCoord(
        ra=tab2["EMU_ra_deg"].values * u.degree,
        dec=tab2["EMU_dec_deg"].values * u.degree,
        frame="icrs",
    )
    names = coords.to_string(style="hmsdms", precision=1, sep="", pad=True)
    tab2["EMU_source_name"] = [
        "EMU J" + name[:-2].replace(" ", "") for name in names
    ]
    return tab2


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_csv_and_fits(df: pd.DataFrame, csv_path: str, fits_cols: list) -> None:
    """Write DataFrame to CSV and a subset to FITS."""
    df.to_csv(csv_path, index=False)
    subset = pd.read_csv(csv_path, usecols=fits_cols, low_memory=False)
    fits_path = csv_path.replace(".csv", ".fits")
    Table.from_pandas(subset).write(fits_path, format="fits", overwrite=True)
    logger.info("Written %s and %s", csv_path, fits_path)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def run_value_added(
    sbid: str,
    output_dir: str,
    wise_catalogs_dir: str,
    desi_catalogs_dir: str,
    cat_version: str = "v1",
) -> None:
    cons_dir = os.path.join(output_dir, sbid, "cons_catalogs")

    # ---- Load consolidated catalogue ---------------------------------------
    cons_csv = os.path.join(cons_dir, f"consolidated_catalog_last{cat_version}.csv")
    if not os.path.exists(cons_csv):
        raise FileNotFoundError(f"Consolidated catalogue not found: {cons_csv}")
    tab = pd.read_csv(cons_csv)
    logger.info("Loaded consolidated catalogue: %d sources", len(tab))

    # ---- Build output schema -----------------------------------------------
    tab2 = build_tab2(tab)

    # ---- CatWISE cross-match -----------------------------------------------
    wise_csv = os.path.join(wise_catalogs_dir, f"{sbid}_catwise_catalog.csv")
    if not os.path.exists(wise_csv):
        raise FileNotFoundError(f"CatWISE catalogue not found: {wise_csv}")
    tab_wise = pd.read_csv(wise_csv)
    tab2 = crossmatch_catwise(tab2, tab_wise)

    # ---- DESI cross-match (optional) ---------------------------------------
    desi_csv = os.path.join(desi_catalogs_dir, f"{sbid}_desi_dr10_catalog.csv")
    if os.path.exists(desi_csv):
        tab_desi = pd.read_csv(desi_csv)
        if len(tab_desi) > 1:
            tab2 = crossmatch_desi(tab2, tab_desi)
            # Photo-z (optional within DESI)
            desi_z_csv = os.path.join(
                desi_catalogs_dir, f"{sbid}_desi_dr10_z_catalog.csv"
            )
            if os.path.exists(desi_z_csv):
                tab_desi_z = pd.read_csv(desi_z_csv)
                if len(tab_desi_z) > 1:
                    tab2 = add_desi_photoz(tab2, tab_desi_z)
            else:
                logger.info("No DESI photo-z catalogue found for SBid %s", sbid)
        else:
            logger.warning("DESI catalogue for SBid %s has ≤1 row, skipping", sbid)
    else:
        logger.info("No DESI catalogue found for SBid %s — skipping DESI step", sbid)

    # ---- Assign IAU names --------------------------------------------------
    tab2 = assign_iau_names(tab2)

    # ---- Write outputs -----------------------------------------------------
    fits_summary_cols = ["EMU_source_name", "EMU_ra_deg", "EMU_dec_deg",
                         "EMU_type", "EMU_score"]

    # All sources
    all_csv = os.path.join(cons_dir, "value_added_catwise_ls_all.csv")
    _write_csv_and_fits(tab2, all_csv, fits_summary_cols)

    # Extended sources only
    ext = tab2[tab2["EMU_type"] != "C"].reset_index(drop=True)
    ext_out = ext.drop(columns=["catwise_separation"], errors="ignore")
    ext_csv = os.path.join(cons_dir, "value_added_catwise_ls_allExtended.csv")
    _write_csv_and_fits(ext_out, ext_csv, fits_summary_cols)

    logger.info("Value-added catalogue complete for SBid %s", sbid)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    try:
        run_value_added(
            sbid=args.SBid,
            output_dir=args.output_dir,
            wise_catalogs_dir=args.wise_catalogs_dir,
            desi_catalogs_dir=args.desi_catalogs_dir,
            cat_version=args.cat_version,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
