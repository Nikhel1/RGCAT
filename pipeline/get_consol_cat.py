#!/usr/bin/env python
"""
get_consol_cat.py
-----------------
Builds the consolidated radio galaxy catalogue for a single EMU tile.

For each Gal-DINO detection above a score threshold the script:
  1. Optionally recalibrates the confidence score using spline calibration.
  2. Determines the tightest bounding box enclosing the detected emission.
  3. Cross-matches selavy components inside that box and consolidates them.
  4. Computes host-galaxy pixel → sky position.
  5. Writes per-class FITS and DS9 region files.

Reads:
  <annotations_dir>/instances_angle_<SBid>.json      (COCO annotation stub)
  <output_dir>/results0_all_<SBid>.json               (Gal-DINO predictions)
  <catalogs_dir>/*<SBid>*.csv                          (selavy CSV)
  <fits_file_dir>/*_emu.fits                           (preprocessed cutouts)

Writes:
  <output_dir>/<SBid>/cons_catalogs/
      consolidated_catalog_v1.csv
      consolidated_catalog_lastv1.csv / .fits
      consolidated_catalog_last_allExtended_v1.fits   (sentinel for pipeline)
      consolidated_catalog_last_<type>v1.fits          (per class)
      ds9_regions_all_v1.reg
      ds9_regions_<type>_v1.reg

Usage:
    python get_consol_cat.py --Image_SBid image.i.EMU_0044-46.SB50180....fits
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import warnings
from collections import Counter
from shutil import copy2
from typing import Dict, Iterator, List, Optional, Tuple, Union

import astropy.units as u
import astropink as pu
import cv2
import numpy as np
import pandas as pd
import torch
from astropy.coordinates import Angle, SkyCoord, search_around_sky
from astropy.io import ascii, fits
from astropy.io.votable import parse as parse_votable
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from pycocotools.coco import COCO
from regions import PixCoord, RectanglePixelRegion
from scipy.special import softmax
from skimage.measure import label
from skimage.morphology import closing, convex_hull_image
from torchvision.ops.boxes import box_area

warnings.simplefilter("ignore", category=AstropyWarning)

# ---------------------------------------------------------------------------
# Logging  (file handler added inside main() once output dir is known)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("get_consol_cat")


# ---------------------------------------------------------------------------
# Spline calibration imports  (path resolved at runtime)
# ---------------------------------------------------------------------------
def _setup_splinecalibration_path(repo_root: str) -> None:
    """Add the splinecalibration package to sys.path."""
    cal_path = os.path.join(repo_root, "splinecalibration")
    if cal_path not in sys.path:
        sys.path.insert(0, cal_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build consolidated radio galaxy catalogue from Gal-DINO outputs."
    )
    parser.add_argument(
        "--Image_SBid",
        type=str,
        required=True,
        help="EMU tile filename (basename), e.g. image.i.EMU_0044-46.SB50180....fits",
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
        "--shared_annotations_dir",
        type=str,
        default="data/annotations",
        help=(
            "Directory containing shared COCO annotation stubs "
            "(train/val calibration JSONs). Per-tile annotation is read "
            "from all_sbs_dir/<SBid>/ (default: data/annotations)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/CatV1",
        help="Base output directory for Gal-DINO results (default: outputs/CatV1)",
    )
    parser.add_argument(
        "--cat_version",
        type=str,
        default="v1",
        help="Version suffix used in output filenames (default: v1)",
    )
    parser.add_argument(
        "--img_pixel_size_arcsec",
        type=float,
        default=2.0,
        help="Pixel scale of the preprocessed cutouts in arcseconds (default: 2.0)",
    )
    parser.add_argument(
        "--no_calibrate_score",
        action="store_true",
        default=False,
        help="Disable spline confidence score recalibration",
    )
    parser.add_argument(
        "--no_get_angles",
        action="store_true",
        default=False,
        help="Disable contour-based bounding box angle refinement",
    )
    return parser


# ---------------------------------------------------------------------------
# Coordinate transformers
# ---------------------------------------------------------------------------
class SkyToPixelTransformer:
    """Convert sky coordinates to pixel offsets relative to a centre."""

    def __init__(
        self,
        center_coord: SkyCoord,
        sky_coords: SkyCoord,
        transform: Tuple[int, float],
        pixel_scale: Union[None, Angle, Tuple[Angle, Angle]] = u.arcsecond,
        wcs: Optional[WCS] = None,
    ) -> None:
        self.center_coord = center_coord
        self.pixel_scale = pixel_scale
        self.transform = transform
        self.wcs = wcs
        self.coords: Dict[str, np.ndarray] = {}
        self.coords["sky"] = sky_coords
        self.coords["offsets-angular"] = center_coord.spherical_offsets_to(sky_coords)

        if wcs is not None:
            self.coords["offsets-pixel"] = self._wcs_skycoord_to_pixels()
        elif pixel_scale is not None:
            self.coords["offsets-pixel"] = self._delta_sky_to_pixels()
        else:
            raise ValueError(
                "Either wcs or pixel_scale must be provided."
            )
        self.coords["offsets-neuron"] = self._spatial_transform_points()

    def _delta_sky_to_pixels(self) -> Tuple:
        offsets = self.coords["offsets-angular"]
        ps = self.pixel_scale
        if not isinstance(ps, tuple):
            ps = (ps, ps)
        scales = tuple(u.pixel_scale(p / u.pixel) for p in ps)
        return (
            -offsets[0].to(u.pixel, scales[0]),
            offsets[1].to(u.pixel, scales[1]),
        )

    def _wcs_skycoord_to_pixels(self) -> Tuple:
        pix = skycoord_to_pixel(self.coords["sky"], self.wcs) * u.pixel
        cen = skycoord_to_pixel(self.center_coord, self.wcs) * u.pixel
        return tuple(p - c for p, c in zip(pix, cen))

    def _spatial_transform_points(self) -> Tuple:
        flip, angle = self.transform
        angle = -angle
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        coords = np.array(self.coords["offsets-pixel"])
        tc = (coords.T @ rot).T
        if flip == 1:
            tc[1] = -tc[1]
        return (tc[0] * u.pixel, tc[1] * u.pixel)


class PixelToSkyTransformer:
    """Convert pixel coordinates to sky positions via a WCS."""

    def __init__(
        self,
        center_coord: SkyCoord,
        pixel_coords: Tuple[np.ndarray, np.ndarray],
        wcs: WCS,
    ) -> None:
        self.coords: Dict[str, np.ndarray] = {}
        self.coords["offsets-pixel"] = pixel_coords
        x, y = pixel_coords
        self.coords["offsets-sky"] = SkyCoord.from_pixel(
            x * u.pixel, y * u.pixel, wcs, origin=1, mode="all"
        )


# ---------------------------------------------------------------------------
# Image processing utilities
# ---------------------------------------------------------------------------
def rms_estimate(
    data: np.ndarray,
    mode: str = "mad",
    clip_rounds: int = 2,
    outlier_thres: float = 3.0,
) -> float:
    data = data[np.isfinite(data)]
    if mode == "mad":
        clip_fn = lambda d: np.median(np.abs(d - np.median(d)))
    elif mode == "std":
        clip_fn = np.std
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    for _ in range(clip_rounds):
        data = data[np.abs(data) < outlier_thres * clip_fn(data)]
    return float(np.std(data))


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------
def bbox_in_center(bbox: list, image_width: int, image_height: int) -> bool:
    """Return True if the bounding box straddles the image centre."""
    x, y, w, h = bbox
    cx, cy = image_width / 2, image_height / 2
    return (x < cx < x + w) and (y < cy < y + h)


def is_box_inside(box: list, central_box: list) -> bool:
    """Return True if *box* lies entirely within *central_box*."""
    x, y, w, h = box
    cx, cy, cw, ch = central_box
    return (
        x >= cx - cw / 2
        and y >= cy - ch / 2
        and x + w <= cx + cw / 2
        and y + h <= cy + ch / 2
    )


# ---------------------------------------------------------------------------
# Score calibration helpers
# ---------------------------------------------------------------------------
def get_scores_and_labels(
    gt_file: str, res_file: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract max-score central predictions and binary correctness labels."""
    gt_coco = COCO(gt_file)
    result_coco = gt_coco.loadRes(res_file)
    scores, labels = [], []

    for image_id in gt_coco.getImgIds():
        iw = gt_coco.imgs[image_id]["width"]
        ih = gt_coco.imgs[image_id]["height"]

        gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=image_id))
        gt_bboxes = [a["bbox"] for a in gt_anns]
        gt_cats   = [a["category_id"] for a in gt_anns]
        cen_gt = [gt_cats[i] for i, b in enumerate(gt_bboxes) if bbox_in_center(b, iw, ih)]
        if not cen_gt:
            continue

        res_anns  = result_coco.loadAnns(result_coco.getAnnIds(imgIds=image_id))
        res_bboxes = [a["bbox"] for a in res_anns]
        res_scores = [a["score"] for a in res_anns]
        res_cats   = [a["category_id"] for a in res_anns]
        cen_ids    = [i for i, b in enumerate(res_bboxes) if bbox_in_center(b, iw, ih)]
        if not cen_ids:
            continue

        cen_scores = [res_scores[i] for i in cen_ids]
        cen_cats   = [res_cats[i] for i in cen_ids]
        best_cat   = cen_cats[int(np.argmax(cen_scores))]

        label = 1.0 if cen_gt[0] == best_cat else 0.0
        scores.append(max(cen_scores))
        labels.append(label)

    return np.array(scores), np.array(labels)


# ---------------------------------------------------------------------------
# DS9 region filter
# ---------------------------------------------------------------------------
def filter_ds9_regions(
    source_filenames: pd.Series, input_file: str, output_file: str
) -> None:
    """Write a DS9 region file containing only sources in *source_filenames*."""
    try:
        regions = ascii.read(input_file, data_start=2, format="no_header")
    except Exception as exc:
        logger.warning("Could not read DS9 region file %s: %s", input_file, exc)
        return

    fnames_set = set(source_filenames)
    filtered = [
        r for r in regions
        if any(r["col5"].split("text=")[1] in fn for fn in fnames_set)
    ]

    with open(output_file, "w") as f:
        f.write("# Region file format: DS9 CARTA 3.0.0\n")
        f.write(
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman"'
            " select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        f.write("fk5\n")
        for r in filtered:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")


# ---------------------------------------------------------------------------
# Catalogue writer
# ---------------------------------------------------------------------------
CATEGORY_IDS = {1: "FR-II", 2: "FR-I", 3: "FR-X", 4: "R", 5: "C", 6: "S"}

def _to_list(x):
    return [x.tolist()]


def write_subset_fits_and_ds9(
    data: pd.DataFrame,
    which_sources: str,
    cat_version: str,
    out_dir_sbid: str,
    filter_col: str,
    filter_val,
    reg_input: str,
) -> None:
    """Filter *data* on one column, write FITS and DS9 subset files."""
    if filter_val == "!= S":  # all extended shorthand
        subset = data[data["type"] != "S"].reset_index(drop=True)
    elif filter_val == "> 1":  # multi-component shorthand
        subset = data[data["n_components"] > 1].reset_index(drop=True)
    else:
        subset = data[data[filter_col] == filter_val].reset_index(drop=True)

    subset_out = subset.drop(columns=["n_components"], errors="ignore").rename(
        columns={"wise_ra_cons": "ra_cons", "wise_dec_cons": "dec_cons"}
    )
    fits_path = os.path.join(
        out_dir_sbid, "cons_catalogs",
        f"consolidated_catalog_last_{which_sources}{cat_version}.fits",
    )
    Table.from_pandas(subset_out).write(fits_path, format="fits", overwrite=True)

    reg_out = os.path.join(
        out_dir_sbid, "region_files", f"ds9_regions_{which_sources}{cat_version}.reg"
    )
    filter_ds9_regions(subset["filename_cons"], reg_input, reg_out)
    logger.info("Written %s FITS (%d sources)", which_sources, len(subset_out))


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def run_consolidation(
    sbid: str,
    emu_tile_path: str,
    shared_annotations_dir: str,
    output_dir: str,
    catalogs_dir: str,
    all_sbs_dir: str = "data/all_SBs",
    cat_version: str = "v1",
    img_pixel_size_arcsec: float = 2.0,
    calibrate_score: bool = True,
    get_angles: bool = True,
    repo_root: str = ".",
) -> None:
    """Run the full consolidation pipeline for one SBid."""

    _setup_splinecalibration_path(repo_root)
    from recalibrate import get_recalibration_function  # noqa: E402

    out_dir_sbid = output_dir
    cons_dir     = os.path.join(out_dir_sbid, "cons_catalogs")
    reg_dir      = os.path.join(out_dir_sbid, "region_files")
    os.makedirs(cons_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)

    # Add file handler so output also goes to a log file
    log_path = os.path.join(cons_dir, f"get_consol_cat{cat_version}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Archive script copy
    copy2(__file__, os.path.join(cons_dir, f"get_consol_cat{cat_version}.py"))

    # ---- Paths -------------------------------------------------------------
    data_keyword = "all"
    gt_file  = os.path.join(all_sbs_dir, sbid, f"instances_angle_{sbid}.json")
    res_file = os.path.join(output_dir, f"results0_{data_keyword}_{sbid}.json")

    catalog_pattern = os.path.join(catalogs_dir, f"*{sbid}*.csv")
    catalog_matches = glob.glob(catalog_pattern)
    if not catalog_matches:
        raise FileNotFoundError(f"No selavy catalog found matching: {catalog_pattern}")
    if len(catalog_matches) > 1:
        logger.warning("Multiple catalogs matched; using %s", catalog_matches[0])
    catalog_file = catalog_matches[0]

    # Preprocessed FITS cutouts
    fits_file_dir = os.path.join(all_sbs_dir, sbid, "All_fits")

    # ---- Load selavy catalog -----------------------------------------------
    tab_selavy = pd.read_csv(catalog_file, header=0)
    logger.info("Selavy catalog: %d components", len(tab_selavy))

    # ---- Score calibration -------------------------------------------------
    frecal = None
    if calibrate_score:
        gt_train = os.path.join(shared_annotations_dir, "instances_train2017.json")
        res_train = os.path.join(output_dir, "results0_train.json")
        gt_val    = os.path.join(shared_annotations_dir, "instances_val2017.json")
        res_val   = os.path.join(output_dir, "results0_val.json")

        scorest, labelst = get_scores_and_labels(gt_train, res_train)
        scoresv, labelsv = get_scores_and_labels(gt_val, res_val)

        spline_method, splines = "natural", 2
        frecal = get_recalibration_function(scorest, labelst, spline_method, splines)
        logger.info("Score recalibration function fitted on %d train samples", len(scorest))

    # ---- Load COCO results -------------------------------------------------
    gt_coco     = COCO(gt_file)
    result_coco = gt_coco.loadRes(res_file)
    image_ids   = gt_coco.getImgIds()

    with open(gt_file) as f:
        coco_data = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # ---- Extract per-image central predictions ----------------------------
    max_boxes_centre     : Dict[int, list]  = {}
    max_scores_centre    : Dict[int, float] = {}
    max_keypoints_centre : Dict[int, list]  = {}
    max_categories_centre: Dict[int, int]   = {}
    null_pred            : List[int]        = []
    null_pred_centre     : List[int]        = []
    small_boxes_in_centre: Dict[int, Optional[list]] = {}

    for image_id in image_ids:
        iw = result_coco.imgs[image_id]["width"]
        ih = result_coco.imgs[image_id]["height"]
        res_anns   = result_coco.loadAnns(result_coco.getAnnIds(imgIds=image_id))
        res_bboxes = [a["bbox"]       for a in res_anns]
        res_scores = [a["score"]      for a in res_anns]
        res_keys   = [a["keypoints"]  for a in res_anns]
        res_cats   = [a["category_id"]for a in res_anns]

        if not res_bboxes:
            logger.debug("No predictions for image_id %d", image_id)
            null_pred.append(image_id)
            continue

        cen_ids = [i for i, b in enumerate(res_bboxes) if bbox_in_center(b, iw, ih)]
        if not cen_ids:
            null_pred_centre.append(image_id)
            # Fall back to highest-score prediction anywhere in image
            cen_ids = list(range(len(res_bboxes)))

        cen_scores = [res_scores[i] for i in cen_ids]
        best       = int(np.argmax(cen_scores))

        max_box   = res_bboxes[cen_ids[best]]
        max_score = cen_scores[best]
        max_key   = res_keys[cen_ids[best]]
        max_cat   = res_cats[cen_ids[best]]

        if calibrate_score and frecal is not None:
            max_score = float(np.clip(frecal(float(max_score)), 0.0, 1.0))

        max_boxes_centre[image_id]      = max_box
        max_scores_centre[image_id]     = max_score
        max_keypoints_centre[image_id]  = max_key
        max_categories_centre[image_id] = max_cat

        # Small boxes fully inside the central box
        inside = [
            res_bboxes[i]
            for i, b in enumerate(res_bboxes)
            if b != max_box
            and is_box_inside(b, max_box)
            and res_scores[i] >= max_score
        ]
        small_boxes_in_centre[image_id] = inside or None

    logger.info(
        "Images: %d total | %d no predictions | %d no central predictions",
        len(image_ids), len(null_pred), len(null_pred_centre),
    )

    # ---- Sort by score and build consolidated catalogue -------------------
    sorted_scores = dict(
        sorted(max_scores_centre.items(), key=lambda kv: kv[1], reverse=True)
    )
    tab           = tab_selavy.copy()
    positions     = SkyCoord(
        tab["ra_deg_cont"].values * u.deg,
        tab["dec_deg_cont"].values * u.deg,
    )
    tab_consolidated = pd.DataFrame(columns=list(tab.columns) + [
        "ra_deg_cons", "dec_deg_cons", "flux_peak_cons", "flux_int_cons",
        "img_id_cons", "filename_cons", "radio_bbox_cons", "PA",
        "n_components", "type", "score", "wise_ra_cons", "wise_dec_cons",
    ])

    # Region file
    reg_file = os.path.join(reg_dir, f"ds9_regions_all{cat_version}.reg")
    reg_f    = open(reg_file, "w")
    reg_f.write("# Region file format: DS9 CARTA 3.0.0\n")
    reg_f.write(
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman"'
        " select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
    )
    reg_f.write("fk5\n")

    score_cut_non_s = 0.3  # minimum score for non-single-source detections

    for _idx, (image_id, _score) in enumerate(sorted_scores.items(), start=1):
        if image_id in null_pred:
            continue

        cat_label = CATEGORY_IDS[max_categories_centre[image_id]]
        if cat_label != "S" and sorted_scores[image_id] <= score_cut_non_s:
            continue

        try:
            filename = id_to_filename[image_id]
            fits_path = os.path.join(fits_file_dir, filename.replace("rgb.png", "emu.fits"))
            with fits.open(fits_path) as dlist:
                d = dlist[0]
                d_wcs  = WCS(d.header).celestial
                d_data = np.squeeze(d.data).copy()

            d_size_arcmin = d_data.shape[0] * img_pixel_size_arcsec / 60.0
            nx, ny = d_data.shape

            # Match selavy components within the cutout footprint
            src_id_str = filename[:-8]  # strip _rgb.png
            tab_idx    = tab[
                tab.apply(
                    lambda r: f"{r['ra_deg_cont']}_{r['dec_deg_cont']}" == src_id_str,
                    axis=1,
                )
            ]
            position  = SkyCoord(
                tab_idx["ra_deg_cont"].values * u.deg,
                tab_idx["dec_deg_cont"].values * u.deg,
            )
            matches   = search_around_sky(
                position, positions, seplimit=d_size_arcmin * u.arcminute
            )
            src_matches = matches[1]
            spatial_pos = SkyToPixelTransformer(
                position,
                positions[src_matches],
                (0, 0.0),
                pixel_scale=img_pixel_size_arcsec * u.arcsecond,
            )
            source_x = np.array(
                spatial_pos.coords["offsets-pixel"][0].value + nx / 2, dtype=int
            )
            source_y = np.array(
                spatial_pos.coords["offsets-pixel"][1].value + ny / 2, dtype=int
            )

            x_min, y_min, width, height = max_boxes_centre[image_id]
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            angle    = 0.0
            pmask_correction = False

            if cat_label == "S":
                reg = RectanglePixelRegion(
                    PixCoord(x=x_center, y=y_center),
                    width=width / 2, height=height / 2,
                    angle=Angle(0, "deg"),
                )
            else:
                reg = RectanglePixelRegion(
                    PixCoord(x=x_center, y=y_center),
                    width=width, height=height,
                    angle=Angle(0, "deg"),
                )
                if get_angles:
                    srcs_in_box_init  = reg.contains(PixCoord(source_x, source_y))
                    srcs_in_box_coord = PixCoord(source_x, source_y)[srcs_in_box_init]
                    srcs_set          = set(zip(srcs_in_box_coord.x, srcs_in_box_coord.y))

                    noise     = rms_estimate(d_data, mode="mad")
                    mask_img  = np.zeros_like(d_data)
                    pmask     = 50  # min pixels to include an island
                    indv_sizes: List[int] = []

                    for mask in pu.island_segmentation(d_data, 5.0 * noise):
                        if np.sum(mask) < pmask:
                            continue
                        argmask  = np.argwhere(mask)
                        argmaskp = PixCoord(argmask[:, 1], argmask[:, 0])
                        if (
                            any(reg.contains(argmaskp))
                            and srcs_set & set(zip(argmaskp.x, argmaskp.y))
                        ):
                            mask_img[mask] = 1
                            indv_sizes.append(int(np.sum(mask)))

                    if np.any(mask_img):
                        contours, _ = cv2.findContours(
                            mask_img.astype(np.uint8),
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        all_contours = np.concatenate(contours)
                        (x_center, y_center), (width, height), angle = (
                            cv2.minAreaRect(all_contours)
                        )
                        reg = RectanglePixelRegion(
                            PixCoord(x=x_center, y=y_center),
                            width=width, height=height,
                            angle=Angle(angle, "deg"),
                        )
                        pmask_correction = True

                        # Refine: if two prominent blobs differ by >2×, use larger
                        if len(indv_sizes) > 1:
                            sorted_sizes = sorted(indv_sizes, reverse=True)
                            pmask = (
                                sorted_sizes[0]
                                if sorted_sizes[0] / sorted_sizes[1] > 2.0
                                else sorted_sizes[1]
                            )
                            logger.debug(
                                "pmask correction applied for image_id %d", image_id
                            )
                            mask_img = np.zeros_like(d_data)
                            for mask in pu.island_segmentation(d_data, 5.0 * noise):
                                if np.sum(mask) < pmask:
                                    continue
                                argmask  = np.argwhere(mask)
                                argmaskp = PixCoord(argmask[:, 1], argmask[:, 0])
                                if (
                                    any(reg.contains(argmaskp))
                                    and srcs_set & set(zip(argmaskp.x, argmaskp.y))
                                ):
                                    mask_img[mask] = 1
                            if np.any(mask_img):
                                contours, _ = cv2.findContours(
                                    mask_img.astype(np.uint8),
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )
                                all_contours = np.concatenate(contours)
                                (x_center, y_center), (width, height), angle = (
                                    cv2.minAreaRect(all_contours)
                                )
                                reg = RectanglePixelRegion(
                                    PixCoord(x=x_center, y=y_center),
                                    width=width, height=height,
                                    angle=Angle(angle, "deg"),
                                )

            srcs_in_box = reg.contains(PixCoord(source_x, source_y))
            matched_idx = list(src_matches[srcs_in_box])
            if not matched_idx:
                logger.debug("image_id %d: all sources already consolidated", image_id)
                continue

            tab1 = tab.iloc[matched_idx]
            if cat_label != "S":
                if len(tab1) == 1 and cat_label != "R":
                    continue

            tab1_c = tab1.apply(_to_list) if len(tab1) > 1 else tab1.copy()

            # Radio sky position from bounding box centre
            sky_r = PixelToSkyTransformer(position, (x_center, y_center), d_wcs)
            sky_coords_r = sky_r.coords["offsets-sky"]

            tab1_c["ra_deg_cons"]    = sky_coords_r.ra.deg
            tab1_c["dec_deg_cons"]   = sky_coords_r.dec.deg
            tab1_c["flux_peak_cons"] = float(tab1["flux_peak"].max())
            tab1_c["flux_int_cons"]  = float(tab1["flux_int"].sum())
            tab1_c["img_id_cons"]    = image_id
            tab1_c["filename_cons"]  = src_id_str
            bbox_val = [None] * len(tab1_c)
            bbox_val[0] = [x_center, y_center, width, height, angle]
            tab1_c["radio_bbox_cons"] = bbox_val
            tab1_c["PA"]              = angle
            tab1_c["n_components"]    = len(tab1)
            tab1_c["type"]            = cat_label
            tab1_c["score"]           = sorted_scores[image_id]

            # IR host position from keypoints
            kp_x, kp_y = (
                np.array(max_keypoints_centre[image_id][0]),
                np.array(max_keypoints_centre[image_id][1]),
            )
            sky_ir = PixelToSkyTransformer(position, (kp_x, kp_y), d_wcs)
            sky_coords_ir = sky_ir.coords["offsets-sky"]
            tab1_c["wise_ra_cons"]  = sky_coords_ir.ra.deg
            tab1_c["wise_dec_cons"] = sky_coords_ir.dec.deg

            if pmask_correction:
                tab1_c["wise_ra_cons"]  = tab1_c["ra_deg_cons"]
                tab1_c["wise_dec_cons"] = tab1_c["dec_deg_cons"]
            if cat_label == "S":
                tab1_c["wise_ra_cons"]  = position.ra.deg[0]
                tab1_c["wise_dec_cons"] = position.dec.deg[0]

            # Region entry
            reg_f.write(
                f"box({(sky_coords_r.ra - 5*u.arcsecond).deg},"
                f" {(sky_coords_r.dec + 5*u.arcsecond).deg},"
                f" {width*img_pixel_size_arcsec}\","
                f" {height*img_pixel_size_arcsec}\","
                f" {angle}) # color=#2EE6D6 width=2 text={src_id_str}\n"
            )

            # Drop used selavy rows, append to consolidated table
            tab = tab.drop(matched_idx).reset_index(drop=True)
            positions = SkyCoord(
                tab["ra_deg_cont"].values * u.deg,
                tab["dec_deg_cont"].values * u.deg,
            )
            tab_consolidated = pd.concat([tab_consolidated, tab1_c], ignore_index=True)

            # Incremental save
            tab1_c.to_csv(
                os.path.join(cons_dir, f"consolidated_catalog{cat_version}.csv"),
                mode="a", header=False, index=False,
            )
            logger.info(
                "[%d] image_id=%d type=%s score=%.3f n_selavy=%d",
                _idx, image_id, cat_label, sorted_scores[image_id], len(tab1),
            )

        except Exception as exc:
            logger.error("Error processing image_id %d: %s", image_id, exc)
            continue

    reg_f.close()

    # ---- Add leftover selavy sources as singles ----------------------------
    tab.to_csv(
        os.path.join(cons_dir, f"catalog_leftover_selavy{cat_version}.csv"), index=False
    )
    tab["wise_ra_cons"]  = tab["ra_deg_cont"]
    tab["wise_dec_cons"] = tab["dec_deg_cont"]
    tab["filename_cons"] = tab["component_name"]
    islands = Counter(tab["island_id"].values)
    tab["n_components"]  = tab.apply(lambda r: islands[r["island_id"]], axis=1)
    tab["type"]          = "S"
    tab["score"]         = 0
    tab["flux_peak_cons"]= tab["flux_peak"]
    tab["flux_int_cons"] = tab["flux_int"]
    tab_consolidated = pd.concat([tab_consolidated, tab], ignore_index=True)

    last_csv = os.path.join(cons_dir, f"consolidated_catalog_last{cat_version}.csv")
    tab_consolidated.to_csv(last_csv, index=False)
    logger.info("Consolidated catalogue: %d total sources", len(tab_consolidated))

    # ---- Write main FITS ---------------------------------------------------
    main_fits_cols = ["filename_cons", "wise_ra_cons", "wise_dec_cons", "type", "score"]
    data_fits = pd.read_csv(last_csv, usecols=main_fits_cols, low_memory=False)
    data_fits.rename(
        columns={"wise_ra_cons": "ra_cons", "wise_dec_cons": "dec_cons"}, inplace=True
    )
    Table.from_pandas(data_fits).write(
        last_csv.replace(".csv", ".fits"), format="fits", overwrite=True
    )

    # ---- Write per-class FITS and DS9 region subsets ----------------------
    data = pd.read_csv(
        last_csv,
        usecols=["filename_cons", "wise_ra_cons", "wise_dec_cons",
                 "type", "score", "n_components"],
        low_memory=False,
    )
    reg_all = os.path.join(reg_dir, f"ds9_regions_all{cat_version}.reg")

    subsets = [
        ("multiSelavyInBbox", "n_components", "> 1"),
        ("allExtended",       "type",         "!= S"),
        ("FR-I",              "type",         "FR-I"),
        ("FR-II",             "type",         "FR-II"),
        ("FR-X",              "type",         "FR-X"),
        ("R",                 "type",         "R"),
        ("C",                 "type",         "C"),
    ]
    for which, col, val in subsets:
        write_subset_fits_and_ds9(
            data, which, cat_version, out_dir_sbid, col, val, reg_all
        )

    logger.info("get_consol_cat complete for SBid %s", sbid)
    logger.removeHandler(fh)
    fh.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    tile_filename = args.Image_SBid
    sbmatch = re.search(r"SB(\d+)", tile_filename)
    if not sbmatch:
        logger.error(
            "Could not extract SBid from '%s'. Expected *SB#####*.fits", tile_filename
        )
        sys.exit(1)
    sbid = sbmatch.group(1)
    logger.info("Working on SBid: %s", sbid)

    emu_tile_path = os.path.join(args.emu_images_dir, tile_filename)
    if not os.path.exists(emu_tile_path):
        logger.error("EMU tile not found: %s", emu_tile_path)
        sys.exit(1)

    # repo root is two levels up from pipeline/
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        run_consolidation(
            sbid=sbid,
            emu_tile_path=emu_tile_path,
            shared_annotations_dir=args.shared_annotations_dir,
            output_dir=args.output_dir,
            catalogs_dir=args.emu_catalogs_dir,
            all_sbs_dir=args.all_sbs_dir,
            cat_version=args.cat_version,
            img_pixel_size_arcsec=args.img_pixel_size_arcsec,
            calibrate_score=not args.no_calibrate_score,
            get_angles=not args.no_get_angles,
            repo_root=repo_root,
        )
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
