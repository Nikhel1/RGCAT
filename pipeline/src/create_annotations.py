"""
create_annotations.py
---------------------
Helper functions for building COCO-format annotation structures.

Used by create_fake_json_All_fits.py to generate the per-tile JSON stubs
required by Gal-DINO inference.

Functions
---------
create_sub_masks            : split a mask image into per-colour submasks
create_sub_mask_annotation  : convert a submask to polygon + segmentation
create_category_annotation  : build the COCO "categories" list
create_image_annotation     : build a single COCO "images" entry
create_annotation_format    : build a single COCO "annotations" entry
get_coco_json_format        : return the empty COCO JSON skeleton
"""

import json
import os

import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure


def create_sub_masks(mask_image: Image.Image, width: int, height: int) -> dict:
    """Split a mask image into per-colour binary submasks.

    Each unique RGB colour in the mask becomes an entry in the returned dict,
    keyed by the string representation of the colour tuple.  One pixel of
    padding is added in each direction so that contour detection does not
    bleed to the image edge.

    Parameters
    ----------
    mask_image : PIL.Image
        RGB mask image.
    width, height : int
        Image dimensions.

    Returns
    -------
    dict[str, PIL.Image]
        Mapping of colour string → 1-bit submask image.
    """
    sub_masks: dict = {}
    for x in range(width):
        for y in range(height):
            pixel = mask_image.getpixel((x, y))[:3]
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                # +2 padding so contours don't touch the border
                sub_masks[pixel_str] = Image.new("1", (width + 2, height + 2))
            sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
    return sub_masks


def create_sub_mask_annotation(sub_mask: np.ndarray):
    """Convert a binary submask to COCO polygon + segmentation lists.

    Parameters
    ----------
    sub_mask : np.ndarray
        2-D boolean / uint8 mask array.

    Returns
    -------
    polygons : list[shapely.Polygon]
    segmentations : list[list[float]]
        Flat coordinate lists suitable for the COCO JSON format.
    """
    contours = measure.find_contours(
        np.array(sub_mask), 0.5, positive_orientation="low"
    )

    polygons = []
    segmentations = []
    for contour in contours:
        # Convert (row, col) → (x, y) and remove padding offset
        for i, (row, col) in enumerate(contour):
            contour[i] = (col - 1, row - 1)

        poly = Polygon(contour).simplify(1.0, preserve_topology=False)
        if poly.is_empty:
            continue

        polygons.append(poly)
        segmentations.append(
            np.array(poly.exterior.coords).ravel().tolist()
        )

    return polygons, segmentations


def create_category_annotation(category_dict: dict) -> list:
    """Build the COCO ``categories`` list from a name→id mapping.

    Parameters
    ----------
    category_dict : dict[str, int]
        E.g. ``{"FR-II": 1, "FR-I": 2, ...}``

    Returns
    -------
    list[dict]
    """
    return [
        {"supercategory": name, "id": cat_id, "name": name}
        for name, cat_id in category_dict.items()
    ]


def create_image_annotation(
    file_name: str, width: int, height: int, image_id: int
) -> dict:
    """Build a single COCO ``images`` entry.

    Parameters
    ----------
    file_name : str
        Basename of the image file.
    width, height : int
        Image dimensions in pixels.
    image_id : int
        Unique integer identifier.

    Returns
    -------
    dict
    """
    return {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id,
    }


def create_annotation_format(
    polygon,
    segmentation: list,
    image_id: int,
    keypoints: list,
    category_id: int,
    annotation_id: int,
) -> dict:
    """Build a single COCO ``annotations`` entry.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        Used to compute the axis-aligned bounding box and area.
    segmentation : list
        Flat coordinate list(s) for the polygon outline.
    image_id : int
    keypoints : list[int]
        Flat [x, y, visibility] triplet.
    category_id : int
    annotation_id : int

    Returns
    -------
    dict
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = width * height  # axis-aligned area (not polygon.area)

    return {
        "segmentation": segmentation,
        "num_keypoints": 1,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "keypoints": keypoints,
        "category_id": category_id,
        "id": annotation_id,
    }


def get_coco_json_format() -> dict:
    """Return an empty COCO JSON skeleton.

    Returns
    -------
    dict
        Standard COCO top-level structure with empty lists/dicts.
    """
    return {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }
