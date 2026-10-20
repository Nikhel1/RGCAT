#!/usr/bin/env python
"""
create_fake_json_All_fits.py
-----------------------------
Generates a COCO-format JSON annotation file for all preprocessed images
of a given EMU tile.  The annotations are "fake" in the sense that they
use a fixed centroid keypoint (image centre) — the model will refine them.

Reads from: all_SBs/<SBid>/All2017/*_rgb.png
Writes to:  annotations/instances_angle_<SBid>.json

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
from PIL import Image
from shapely.geometry import MultiPolygon

from src.create_annotations import (
    create_annotation_format,
    create_category_annotation,
    create_image_annotation,
    create_sub_mask_annotation,
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
        "--images_base_dir",
        type=str,
        default="./EMUcatalog_main/images",
        help="Base directory containing EMU FITS tiles",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="./Allmask/",
        help="Directory containing the reference mask PNGs",
    )
    return parser


# ---------------------------------------------------------------------------
# Annotation builder
# ---------------------------------------------------------------------------
def build_images_and_annotations(
    mask_path: str,
    image_path: str,
) -> tuple[list, list, int]:
    """
    Build COCO 'images' and 'annotations' lists.

    Uses a single representative mask image to derive a shared annotation
    (fake centroid keypoint at image centre) that is replicated for every
    source image under *image_path*.

    Returns
    -------
    images, annotations, annotation_count
    """
    annotation_id = 1
    image_id = 1

    # ---- Load reference mask -----------------------------------------------
    mask_files = glob.glob(os.path.join(mask_path, "*_rgb.png"))
    if not mask_files:
        raise FileNotFoundError(
            f"No *_rgb.png mask files found in {mask_path}"
        )
    mask_file = mask_files[0]
    logger.info("Using reference mask: %s", mask_file)

    mask_image_open = Image.open(mask_file).convert("RGB")
    w, h = mask_image_open.size
    ir_x = w / 2.0
    ir_y = h / 2.0
    mask_np = np.array(mask_image_open)

    # Build one annotation from the reference mask
    annotation = None
    for instance_id in np.unique(mask_np[:, :, 1]):
        if instance_id == 0:
            continue
        sub_mask = mask_np[:, :, 1] == instance_id
        unique_classes = np.unique(sub_mask * mask_np[:, :, 2])
        # skip if only background (0)
        non_zero = unique_classes[unique_classes != 0]
        if len(non_zero) == 0:
            continue
        category_id = int(non_zero[0])

        try:
            polygons, segmentations = create_sub_mask_annotation(sub_mask)
        except Exception as exc:
            logger.warning("Skipping instance %d in %s: %s", instance_id, mask_file, exc)
            continue

        keypoints = [round(ir_x), round(ir_y), 2]

        if category_id in MULTIPOLYGON_IDS:
            multi_poly = MultiPolygon(polygons)
            annotation = create_annotation_format(
                multi_poly, segmentations, image_id, keypoints, category_id, annotation_id
            )
        else:
            for poly in polygons:
                seg = [np.array(poly.exterior.coords).ravel().tolist()]
                annotation = create_annotation_format(
                    poly, seg, image_id, keypoints, category_id, annotation_id
                )
        break  # one annotation template is enough

    if annotation is None:
        raise RuntimeError(
            f"Could not derive any annotation from mask at {mask_path}"
        )

    # ---- Iterate over source images ----------------------------------------
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

        ann = dict(annotation)  # copy, then stamp with correct ids
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
    mask_path: str,
) -> None:
    image_path = f"./all_SBs/{sbid}/All2017/"
    out_dir = "./annotations"
    out_file = os.path.join(out_dir, f"instances_angle_{sbid}.json")

    os.makedirs(out_dir, exist_ok=True)

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(CATEGORY_IDS)

    logger.info("Building annotations for SBid %s", sbid)
    coco_format["images"], coco_format["annotations"], annotation_cnt = \
        build_images_and_annotations(mask_path, image_path)

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
        create_annotation_json(sbid=sbid, mask_path=args.mask_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
