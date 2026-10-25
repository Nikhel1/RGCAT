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

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

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
    return parser


# ---------------------------------------------------------------------------
# Synthetic annotation template
# ---------------------------------------------------------------------------
def _make_template_annotation(
    cutout_size: int,
    image_id: int = 1,
    annotation_id: int = 1,
    category_id: int = 6,  # S (Single) — a safe placeholder class
) -> dict:
    """
    Build a fake COCO annotation without needing a reference mask PNG.

    The annotation is a small square centred on the image, which is all
    the downstream Gal-DINO inference code requires as a stub.
    """
    cx, cy = cutout_size / 2.0, cutout_size / 2.0
    half = cutout_size / 8.0  # 1/4 of the image width

    poly = Polygon([
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ])
    segmentation = [np.array(poly.exterior.coords).ravel().tolist()]
    keypoints = [round(cx), round(cy), 2]

    multi_poly = MultiPolygon([poly])
    return create_annotation_format(
        multi_poly, segmentation, image_id, keypoints, category_id, annotation_id
    )


# ---------------------------------------------------------------------------
# Annotation builder
# ---------------------------------------------------------------------------
def build_images_and_annotations(
    image_path: str,
    cutout_size: int = 240,
) -> tuple[list, list, int]:
    """
    Build COCO 'images' and 'annotations' lists without a reference mask.

    A synthetic centred-square annotation template is replicated for every
    source image under *image_path*.

    Returns
    -------
    images, annotations, annotation_count
    """
    annotation_id = 1
    image_id = 1

    template = _make_template_annotation(cutout_size, image_id, annotation_id)
    w = h = cutout_size

    images = []
    annotations = []

    image_files = sorted(glob.glob(os.path.join(image_path, "*rgb.png")))
    if not image_files:
        logger.warning("No *rgb.png images found in %s", image_path)

    for img_file in image_files:
        image = create_image_annotation(
            os.path.basename(img_file), w, h, image_id
        )
        images.append(image)

        ann = dict(template)
        ann["image_id"] = image_id
        ann["id"] = annotation_id
        annotations.append(ann)

        annotation_id += 1
        image_id += 1

    return images, annotations, annotation_id


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def create_annotation_json(
    sbid: str,
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
        build_images_and_annotations(image_path, cutout_size)

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

    try:
        create_annotation_json(
            sbid=sbid,
            all_sbs_dir=args.all_sbs_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
