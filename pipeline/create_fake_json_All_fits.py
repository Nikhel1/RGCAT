#!/usr/bin/env python
"""
create_fake_json_All_fits.py
-----------------------------
Generates a COCO-format JSON annotation file for all preprocessed images
of a given EMU tile.  The annotations are "fake" in the sense that they
use a fixed centroid keypoint (image centre) — the model will refine them.

Reads from: all_SBs/<SBid>/All2017/*_rgb.png
Writes to:  all_SBs/<SBid>/instances_angle_<SBid>.json

Usage (from within CatalogNET_V1/):
    python create_fake_json_All_fits.py --Image_SBid image.i.EMU_0044-46.SB50180....fits
"""

import argparse
import glob
import json
import logging
import os
import re
import sys


from src.create_annotations import (
    create_annotation_format,
    create_category_annotation,
    create_image_annotation,
    get_coco_json_format,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("create_fake_json")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORY_IDS = {
    "FR-II": 1,
    "FR-I":  2,
    "FR-X":  3,
    "R":     4,
    "C":     5,
    "S":     6,
}

# All classes are treated as (possibly) multi-polygon
MULTIPOLYGON_IDS = list(range(1, 8))

# Fixed template annotation body — derived from the reference mask PNG used in
# the original create_fake_json_All_fits.py (J210342.7-554258_rgb.png).
# Every source gets the same stub annotation; Gal-DINO replaces these at
# inference time.  Values must stay constant to produce identical JSON output.
_TEMPLATE_SEGMENTATION = [[
    138.0, 136.5, 133.0, 135.5, 124.0, 128.5, 107.0, 125.5, 100.5, 116.0,
    96.5,  113.0, 95.5,  107.0, 97.5,  103.0, 103.5, 98.0,  101.5, 95.0,
    101.5, 90.0,  103.0, 88.5,  107.0, 88.5,  109.5, 91.0,  109.5, 99.0,
    116.5, 103.0, 118.5, 109.0, 117.5, 113.0, 124.0, 120.5, 132.0, 121.5,
    135.0, 119.5, 141.5, 120.0, 143.5, 128.0, 140.5, 135.0, 138.0, 136.5,
]]
_TEMPLATE_BBOX         = [95.5, 88.5, 48.0, 48.0]
_TEMPLATE_AREA         = 2304.0
_TEMPLATE_CATEGORY_ID  = 1
_TEMPLATE_KEYPOINTS    = [120, 120, 2]
_TEMPLATE_NUM_KP       = 1
_TEMPLATE_ISCROWD      = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fake COCO annotation JSON for a processed EMU tile."
    )
    parser.add_argument(
        "--Image_SBid",
        type=str,
        required=True,
        help="Filename of the EMU FITS tile (basename only)",
    )
    parser.add_argument(
        "--all_sbs_dir",
        type=str,
        default="data/all_SBs",
        help="Working area for per-SBid cutouts (default: data/all_SBs)",
    )
    parser.add_argument(
        "--emu_catalogs_dir",
        type=str,
        default="data/catalogs",
        help="Directory containing selavy CSV catalogs (default: data/catalogs)",
    )
    return parser


# ---------------------------------------------------------------------------
# Annotation builder
# ---------------------------------------------------------------------------
def build_images_and_annotations(
    image_path: str,
    catalog_path: str,
    cutout_size: int = 240,
) -> tuple[list, list, int]:
    """
    Build COCO 'images' and 'annotations' lists.

    Images are ordered by catalog row order (matching the original pipeline's
    glob-over-filesystem behaviour where files were written in catalog order).
    Every source gets the same fixed stub annotation derived from the original
    reference mask PNG — Gal-DINO replaces these at inference time.

    Parameters
    ----------
    image_path   : directory containing *_rgb.png files
    catalog_path : path to the selavy CSV catalog for this tile
    cutout_size  : expected width/height of each cutout image

    Returns
    -------
    images, annotations, annotation_count
    """
    import pandas as pd

    # Load catalog to get canonical source ordering
    tab = pd.read_csv(catalog_path, header=0)
    # Support both old and new column naming conventions
    if "ra_deg_cont" not in tab.columns and "ra_deg_radio" in tab.columns:
        tab.rename(
            columns={"ra_deg_radio": "ra_deg_cont", "dec_deg_radio": "dec_deg_cont"},
            inplace=True,
        )

    w = h = cutout_size
    annotation_id = 1
    image_id = 1
    images = []
    annotations = []

    for _, row in tab.iterrows():
        fname = f"{row['ra_deg_cont']}_{row['dec_deg_cont']}_rgb.png"
        if not os.path.exists(os.path.join(image_path, fname)):
            logger.debug("Skipping missing image: %s", fname)
            continue

        images.append(create_image_annotation(fname, w, h, image_id))

        ann = {
            "segmentation":   _TEMPLATE_SEGMENTATION,
            "num_keypoints":  _TEMPLATE_NUM_KP,
            "area":           _TEMPLATE_AREA,
            "iscrowd":        _TEMPLATE_ISCROWD,
            "image_id":       image_id,
            "bbox":           _TEMPLATE_BBOX,
            "keypoints":      _TEMPLATE_KEYPOINTS,
            "category_id":    _TEMPLATE_CATEGORY_ID,
            "id":             annotation_id,
        }
        annotations.append(ann)

        annotation_id += 1
        image_id += 1

    if not images:
        logger.warning("No matching images found in %s", image_path)

    return images, annotations, annotation_id


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def create_annotation_json(
    sbid: str,
    catalog_path: str,
    all_sbs_dir: str = "data/all_SBs",
    cutout_size: int = 240,
) -> None:
    sb_dir     = os.path.join(all_sbs_dir, sbid)
    image_path = os.path.join(sb_dir, "All2017")
    out_file   = os.path.join(sb_dir, f"instances_angle_{sbid}.json")

    os.makedirs(sb_dir, exist_ok=True)

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(CATEGORY_IDS)

    logger.info("Building annotations for SBid %s", sbid)
    coco_format["images"], coco_format["annotations"], annotation_cnt = \
        build_images_and_annotations(image_path, catalog_path, cutout_size)

    with open(out_file, "w") as f:
        json.dump(coco_format, f)

    logger.info(
        "Created %d annotations for %d images → %s",
        annotation_cnt - 1,
        len(coco_format["images"]),
        out_file,
    )


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

    # Locate catalog
    pattern = os.path.join(args.emu_catalogs_dir, f"*{sbid}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        logger.error("No catalog CSV found matching: %s", pattern)
        sys.exit(1)
    if len(matches) > 1:
        logger.warning("Multiple catalogs match SBid %s — using: %s", sbid, matches[0])
    catalog_path = matches[0]
    logger.info("Found catalog: %s", catalog_path)

    try:
        create_annotation_json(
            sbid=sbid,
            catalog_path=catalog_path,
            all_sbs_dir=args.all_sbs_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
