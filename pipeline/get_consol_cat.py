from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, Union, Dict, Iterator
import warnings
import sys
from shutil import copy2
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval
from skimage.morphology import closing, convex_hull_image
from skimage.measure import label
#from scipy import ndimage
from torchvision.ops.boxes import box_area

import functools
import os
import re
import glob
import cv2
import yaml
import matplotlib.pyplot as plt
import pickle
from scipy.special import softmax
sys.path.append('./splinecalibration')
from cal_metrics.ECE import _ECELoss
from cal_metrics.MCE import _MCELoss
from cal_metrics.KS import plot_KS_graphs
from log_utils import ResultsLog
import utilities as utils
from recalibrate import *

from astropy.io import fits
from astropy.io.votable import parse
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, Angle, search_around_sky
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy.table import Table
import astropink as pu
from regions import PixCoord, RectanglePixelRegion
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Get consolidated Catalogs', add_help=False)
    parser.add_argument('--Image_SBid', type=str, default='image.i.EMU_0044-46.SB50180.cont.taylor.0.restored.conv.fits')
    return parser

parser = argparse.ArgumentParser('Get consolidated Catalogs', parents=[get_args_parser()])
args = parser.parse_args()

# path to annotations file and result file
# To generate results0.json change:
# change: is_test = True 
# In engine.py change line 367: if s>0.05
emu_tile = f"./CatalogNET_V1/EMUcatalog_main/images/{args.Image_SBid}"
SBmatch = re.search(r'SB(\d+)', emu_tile)
SBid = SBmatch.group(1)
in_dir = "CatalogNET_V1/annotations"
out_dir = "outputs_gal/CatV1/v1_without_angle/"
fits_file_dir = f"CatalogNET_V1/all_SBs/{SBid}/All_fits/"
data_keyword = 'all'  # train/val/test/all
gt_file = f"{in_dir}/instances_angle_{SBid}.json" 
res_file = f"outputs_gal/CatV1/v1_without_angle/results0_{data_keyword}_{SBid}.json"
catalogs_dir = os.path.dirname(emu_tile).replace('images', 'catalogs')
search_pattern = os.path.join(catalogs_dir, f'*{SBid}*.csv')
catalog_file = glob.glob(search_pattern)[0]
#catalog_file = "/home/gup039/bowen/work/all_data/EMU_SOM/Data/big_emu/emu_with_catwise2020_desdr1_scos_desi_nocutoff.xml"
img_pixel_size_arcsec = 2.
cat_version = 'v1'
calibrate_score = True
get_angles = True

if not os.path.exists(f"{out_dir}{SBid}"):
    os.mkdir(f"{out_dir}{SBid}")
if not os.path.exists(f"{out_dir}{SBid}/cons_catalogs"):
    os.mkdir(f"{out_dir}{SBid}/cons_catalogs")
old_stdout = sys.stdout
log_file = open(f"{out_dir}{SBid}/cons_catalogs/get_consol_cat{cat_version}.log","w")
sys.stdout = log_file
copy2('get_consol_cat.py', f'{out_dir}/{SBid}/cons_catalogs/get_consol_cat{cat_version}.py')

####################################################################
########## Get Selavy Table
####################################################################

def votable_to_pandas(votable_file):
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()

#votable = parse()
#table = votable.to_table()
#tab_selavy = votable_to_pandas(catalog_file)
tab_selavy = pd.read_csv(catalog_file, header=0, skiprows=0)


####################################################################
########## Some classes and functions 
####################################################################

class SKyToPixelTransformer:
    def __init__(
        self,
        center_coord: SkyCoord,
        sky_coords: SkyCoord,
        transform: Tuple[int, float],
        pixel_scale: Union[None, Angle, Tuple[Angle, Angle]] = u.arcsecond,
        wcs: Union[None, WCS] = None,
        src_idx: np.ndarray = None,
    ):
        self.center_coord = center_coord
        self.pixel_scale = pixel_scale
        self.transform = transform
        self.wcs = wcs
        self.src_idx = src_idx

        self.coords: Dict[str, np.ndarray] = {}
        self.coords["sky"] = sky_coords
        self.coords["offsets-angular"] = self.calculate_spherical_offsets()

        if self.wcs is not None:
            self.coords["offsets-pixel"] = self.wcs_skycoord_to_pixels()
        elif self.pixel_scale is not None:
            self.coords["offsets-pixel"] = self.delta_sky_to_pixels()
        else:
            raise ValueError("Missing WCS and pixel scale information. At least one has to be provided.")

        self.coords["offsets-neuron"] = self.spatial_transform_points()

    def calculate_spherical_offsets(self) -> Tuple[Angle, Angle]:
        return self.center_coord.spherical_offsets_to(self.coords["sky"])

    def delta_sky_to_pixels(self) -> Tuple[Angle, Angle]:
        offsets = self.coords["offsets-angular"]

        if self.pixel_scale is None:
            return offsets

        pixel_scale = (
            self.pixel_scale if isinstance(self.pixel_scale, tuple) else (self.pixel_scale, self.pixel_scale)
        )
        pixel_scale = tuple([u.pixel_scale(ps / u.pixel) for ps in pixel_scale])

        offsets = (
            -offsets[0].to(u.pixel, pixel_scale[0]),
            offsets[1].to(u.pixel, pixel_scale[1]),
        )

        return offsets

    def wcs_skycoord_to_pixels(self) -> Tuple[Angle, Angle]:
        pix = skycoord_to_pixel(self.coords["sky"], self.wcs) * u.pixel
        center_pix = skycoord_to_pixel(self.center_coord, self.wcs) * u.pixel

        pix = [p - c for p, c in zip(pix, center_pix)]

        return pix

    def spatial_transform_points(self) -> Tuple[Angle, Angle]:
        flip, angle = self.transform
        angle = -angle  # Rotate clockwise to match PINK

        rotation = np.array(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )

        coords = np.array(self.coords["offsets-pixel"])
        transform_coords = (coords.T @ rotation).T

        if flip == 1:
            transform_coords[1] = -transform_coords[1]

        return (transform_coords[0] * u.pixel, transform_coords[1] * u.pixel)


class PixelToSkyTransformer:
    def __init__(
        self,
        center_coord: SkyCoord,
        pixel_coords: Tuple[np.ndarray, np.ndarray],
        wcs: WCS
    ):
        self.center_coord = center_coord
        self.pixel_coords = pixel_coords
        self.wcs = wcs

        self.coords: Dict[str, np.ndarray] = {}
        self.coords["offsets-pixel"] = pixel_coords
        self.coords["offsets-sky"] = self.pixel_to_skycoord()

    def pixel_to_skycoord(self) -> SkyCoord:
        x_pixels, y_pixels = self.coords["offsets-pixel"]
        sky_coords = SkyCoord.from_pixel(x_pixels * u.pixel, y_pixels * u.pixel, self.wcs, origin=1, mode='all')
        #sky_coords += self.center_coord
        return sky_coords


def rms_estimate(
    data: np.ndarray,
    mode: str = "mad",
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
    nan_check: bool = True,
    ) -> float:
    
    if bin_perc > 1.0:
        bin_perc /= 100.0

    if mode == "std":
        clipping_func = lambda data: np.std(data)

    elif mode == "mad":
        clipping_func = lambda data: np.median(np.abs(data - np.median(data)))

    else:
        raise ValueError(
            f"{mode} not supported as a clipping mode, available modes are `std` and `mad`. "
        )

    if nan_check:
        data = data[np.isfinite(data)]

    for i in range(clip_rounds):
        data = data[np.abs(data) < outlier_thres * clipping_func(data)]

    noise = np.std(data) #fwhm / 2.355

    return noise


def minmax(data: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Apply MinMax normalistion on a dataset. If `mask` is supplied, derive the statistics from the valid pixels   
    
    
    Arguments:
        data {np.ndarray} -- Data to normalise
    
    Keyword Arguments:
        mask {np.ndarray} -- Boolean mask array to select meaningful data (default: {None})
    
    Returns:
        np.ndarray -- Scaled data
    """
    if mask is not None:
        assert (
            data.shape == mask.shape
        ), "Data and Mask arrays must be of the same shape. "
        d_min = np.nanmin(data[mask])
        d_max = np.nanmax(data[mask])
    else:
        d_min = np.nanmin(data)
        d_max = np.nanmax(data)

    scaled = (data - d_min) / (d_max - d_min)

    return scaled
    

def island_segmentation(
    data: np.ndarray,
    threshold: float,
    return_background: bool = False,
    minimum_island_size: int = 5,
) -> Iterator[np.ndarray]:
    """Yield a set of masks that denote unique islands in a image after a threshold operation
    has been applied. 
    
    Arguments:
        data {np.ndarray} -- Data to produce a set of islands
        threshold {float} -- Threshold level to create the set of islands

    Keyword Arguments:
        return_background {bool} -- Return the zero-index background region determined by scikit-img (default: {False}) 
        minimum_island_size {int} -- The minimum number of pixels required for an island to be returned. Useful to avoid noise peaks creeping through (default: {5})

    Returns:
        Iterator[np.ndarray] -- Set of island masks
    """
    mask = closing(data > threshold)

    img_labels, no_labels = label(mask, return_num=True)

    for i in range(no_labels):
        if i == 0 and return_background is False:
            continue

        island_mask = img_labels == i
        if np.sum(island_mask) <= minimum_island_size:
            continue
        yield island_mask
        

def radio_preprocess(data, pmask=2, lower=3, outer_pixels=True):
    nx, ny = data.shape
    noise = rms_estimate(data.flatten(), mode='mad', clip_rounds=2)
    if np.isnan(noise):
        noise = rms_estimate(data.flatten(), mode='std', clip_rounds=2)

    img_scale = np.zeros_like(data)
    for i, mask in enumerate(island_segmentation(data, lower*noise)):
        if np.sum(mask) <= pmask:
            continue
        
        argmask = np.argwhere(mask==True)
        if any(argmask[:,0] == ny/2) and any(argmask[:,1] == nx/2):
            img_scale[mask] = minmax(np.log10(data[mask]))        

        if outer_pixels:
            img_scale[mask] = minmax(np.log10(data[mask]))
        
        img_scale[~np.isfinite(img_scale)] = 0   # to avoid nans in the image

    return img_scale.astype(np.float32)


def wise_preprocess(data, wpmask=2):
    nx, ny = data.shape
    med = np.median(data.flatten())
    mad = np.median(np.abs(data.flatten() - med))
    
    noise = med + 3*mad
    
    img_scale = np.zeros_like(data)
    for i, mask in enumerate(island_segmentation(data, noise)):
        argmask = np.argwhere(mask==True)
        if np.sum(mask) <= wpmask:
            continue
        img_scale[mask] = minmax(np.log10(data[mask]))
    
    return img_scale.astype(np.float32)


# Function to check if a bounding box is in the center of the image
def bbox_in_center(bbox, image_width, image_height):
    # Calculate the center coordinates of the image
    image_x_center = image_width / 2
    image_y_center = image_height / 2  
    if (
    	(bbox[0] < image_x_center) and  (bbox[0] + bbox[2] > image_x_center) and
    	(bbox[1] < image_y_center) and  (bbox[1] + bbox[3] > image_y_center)
    ):
        return True
    else:
        return False


# Check if any of the boxes in an image are inside the central box
# to avoid consolidating extra point sources near complex sources
def is_box_inside(box, central_box):
    x_min, y_min, width, height = box
    x_center, y_center = x_min + width/2, y_min + height/2
    central_xmin, central_ymin, central_width, central_height = central_box
    central_x, central_y = central_xmin + central_width/2, central_ymin + central_height/2
    # Calculate the coordinates of the corners of the box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2    
    # Check if the box is inside the central_box (excluding the central_box itself)
    return (
        x1 >= central_x - central_width / 2 and
        y1 >= central_y - central_height / 2 and
        x2 <= central_x + central_width / 2 and
        y2 <= central_y + central_height / 2
    )

# Define a custom function to transform each column into a list
def to_list(x):
    return [x.tolist()]
    
# Segregate ds9 files for different types of sources
def filter_ds9_regions(source_filename, input_file, output_file):
    # Read the DS9 region file
    regions = ascii.read(input_file, data_start=2, format='no_header')

    # Create an empty list to store the filtered regions
    filtered_regions = []

    # Loop through each region and check if it matches the specified RA and Dec
    for region in regions:
        parts = region['col5'].split('text=')[1]
        is_present = any(parts in filename for filename in source_filename)
        if is_present:
            filtered_regions.append(region)

    # Write the filtered regions to the output DS9 region file
    with open(output_file, 'w') as f:
        f.write("# Region file format: DS9 CARTA 3.0.0\n")
        f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        f.write("fk5\n")
        for region in filtered_regions:
            r = str(region[0])+','+str(region[1])+','+str(region[2])+','+str(region[3])+','+str(region[4])
            f.write(r + '\n')

####################################################################
########## Calibrate score on validation set and try on test set
####################################################################
if calibrate_score:
	def get_scores_and_labels(gt_file, res_file):
		gt_coco = COCO(gt_file)
		result_coco = gt_coco.loadRes(res_file)
		scores = []
		labels = []

		for image_id in gt_coco.getImgIds():
			image_width = gt_coco.imgs[image_id]['width'] 
			image_height = gt_coco.imgs[image_id]['height']
			gt_annotations = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=image_id))
			gt_categories = [ann['category_id'] for ann in gt_annotations]
			gt_bboxes = [ann['bbox'] for ann in gt_annotations]
			if not (len(gt_bboxes)):
				continue
			gt_box_cen_ids = []
			for gt_i in range (len(gt_bboxes)):
				if not bbox_in_center(gt_bboxes[gt_i], image_width, image_height):
					continue
				else:
					gt_box_cen_ids.append(gt_i)
			gt_central_categories = [gt_categories[i] for i in gt_box_cen_ids]  
			if not (len(gt_central_categories)):
				continue              
		
			result_annotations = result_coco.loadAnns(result_coco.getAnnIds(imgIds=image_id))
			result_scores = [ann['score'] for ann in result_annotations]
			result_categories = [ann['category_id'] for ann in result_annotations]
			result_bboxes = [ann['bbox'] for ann in result_annotations]
			if not (len(result_bboxes)):
				continue        
			res_box_cen_ids = []
			for res_i in range (len(result_bboxes)):
				if not bbox_in_center(result_bboxes[res_i], image_width, image_height):
					continue
				else:
					res_box_cen_ids.append(res_i)        
			result_central_categories = [result_categories[i] for i in res_box_cen_ids]
			result_central_scores = [result_scores[i] for i in res_box_cen_ids]  
			if not (len(result_central_categories)):
				continue     
			max_cats_s_c = result_central_categories[np.where(result_central_scores==np.max(result_central_scores))[0][0]]

			# Check if gt category matches result category
			label = 1.0 if gt_central_categories[0] == max_cats_s_c else 0.0
			scores.append(np.max(result_central_scores))
			labels.append(label)
		return np.array(scores), np.array(labels)

	gt_file_t = f"{in_dir}/instances_train2017.json" 
	res_file_t = f"{out_dir}/results0_train.json"
	scorest, labelst = get_scores_and_labels(gt_file_t, res_file_t)

	gt_file_v = f"{in_dir}/instances_val2017.json" 
	res_file_v = f"{out_dir}/results0_val.json"
	scoresv, labelsv = get_scores_and_labels(gt_file_v, res_file_v)

	gt_file_te = f"{in_dir}/instances_test2017.json" 
	res_file_te = f"{out_dir}/results0_test.json"
	scoreste, labelste = get_scores_and_labels(gt_file_te, res_file_te)

	outdir_cal = f"{out_dir}/{SBid}/score_calibration"
	if not os.path.exists(outdir_cal):
		os.mkdir(outdir_cal)
	# Get recalibration function, based on scores1
	spline_method='natural'; splines=2
	frecal = get_recalibration_function (scorest, labelst, spline_method, splines)
	# Recalibrate
	scorest_cal = np.array([frecal(float(sc)) for sc in scorest])
	scorest_cal[scorest_cal < 0.0] = 0.0; scorest_cal[scorest_cal > 1.0] = 1.0
	scoresv_cal = np.array([frecal(float(sc)) for sc in scoresv])
	scoresv_cal[scoresv_cal < 0.0] = 0.0; scoresv_cal[scoresv_cal > 1.0] = 1.0
	scoreste_cal = np.array([frecal(float(sc)) for sc in scoreste])
	scoreste_cal[scoreste_cal < 0.0] = 0.0; scoreste_cal[scoreste_cal > 1.0] = 1.0
	# combine val and test data
	scoresa = np.concatenate([scoresv, scoreste])
	labelsa = np.concatenate([labelsv, labelste])
	scoresa_cal = np.array([frecal(float(sc)) for sc in scoresa])
	scoresa_cal[scoresa_cal < 0.0] = 0.0; scoresa_cal[scoresa_cal > 1.0] = 1.0

	do_KS_graphs = True
	if do_KS_graphs:
		plot_KS_graphs (scorest, labelst, spline_method, splines, outdir_cal, 'uncalibrated_train', title="Uncalibrated")
		plot_KS_graphs (scorest_cal, labelst, spline_method, splines, outdir_cal, 'calibrated_train', title="Calibrated")
		plot_KS_graphs (scoresv, labelsv, spline_method, splines, outdir_cal, 'uncalibrated_val', title="Uncalibrated")
		plot_KS_graphs (scoresv_cal, labelsv, spline_method, splines, outdir_cal, 'calibrated_val', title="Calibrated")
		plot_KS_graphs (scoreste, labelste, spline_method, splines, outdir_cal, 'uncalibrated_test', title="Uncalibrated")
		plot_KS_graphs (scoreste_cal, labelste, spline_method, splines, outdir_cal, 'calibrated_test', title="Calibrated")
		plot_KS_graphs (scoresa, labelsa, spline_method, splines, outdir_cal, 'uncalibrated_val_test', title="Uncalibrated")
		plot_KS_graphs (scoresa_cal, labelsa, spline_method, splines, outdir_cal, 'calibrated_val_test', title="Calibrated")

	do_score_plots = False
	if do_score_plots:
		# Create a figure and a 3x1 grid of subplots
		fig, axs = plt.subplots(3, 1, figsize=(8, 12))
		# Panel 1: scorest and scorest_cal arrays
		axs[0].hist([scorest, scorest_cal], bins=30, label=['Uncalibrated', 'Calibrated'])
		axs[0].set_title('Training Set')
		axs[0].legend()
		# Panel 2: scoresv and scoresv_cal arrays
		axs[1].hist([scoresv, scoresv_cal], bins=30, label=['Uncalibrated', 'Calibrated'])
		axs[1].set_title('Validation Set')
		axs[1].legend()
		# Panel 3: scoreste and scoreste_cal arrays
		axs[2].hist([scoreste, scoreste_cal], bins=30, label=['Uncalibrated', 'Calibrated'])
		axs[2].set_title('Test Set')
		axs[2].legend()
		# Adjust layout to prevent clipping of titles
		plt.tight_layout()
		# Show the plot
		plt.show()

####################################################################
########## Start here
####################################################################

# Load ground truth annotations
gt_coco = COCO(gt_file)

# Load result annotations
result_coco = gt_coco.loadRes(res_file)

# Get image IDs from the ground truth dataset
image_ids = gt_coco.getImgIds()

# Get filenames
with open(gt_file, 'r') as file:
    coco_data = json.load(file)

# Create a dictionary to map image IDs to file names
image_id_to_filename = {}

# Iterate through images and record the file names
for image in coco_data['images']:
    image_id_to_filename[image['id']] = image['file_name']

# Category Id dictionary
category_ids = {
    1: "FR-II",
    2: "FR-I",
    3: "FR-X",
    4: "R",
    5: "C",
    6: "S"
}

####################################################################
########## Get basic stats, boxes, keypoints etc.
####################################################################

#max_boxes = []
#max_scores = []
#max_keypoints = []
max_boxes_centre = {}
max_scores_centre = {}
max_keypoints_centre = {}
max_categories_centre = {}
Null_pred_boxes = []
Null_pred_boxes_centre = []
small_box_in_max_boxes_centre = {}
small_keys_in_max_boxes_centre = {}
for img_idx in range(len(image_ids)):
	# Choose an image ID
	image_id = image_ids[img_idx]  

	# Get the image dimensions (width and height)
	image_width = result_coco.imgs[image_id]['width'] 
	image_height = result_coco.imgs[image_id]['height']

	# Get result annotations for the same image
	result_annotations = result_coco.loadAnns(result_coco.getAnnIds(imgIds=image_id))

	# Extract bounding boxes from result annotations
	result_bboxes = [ann['bbox'] for ann in result_annotations]
	result_scores = [ann['score'] for ann in result_annotations]
	result_keypoints = [ann['keypoints'] for ann in result_annotations]
	result_categories = [ann['category_id'] for ann in result_annotations]

	if not (len(result_bboxes)):
		print('No Pred boxes for image_id:', image_id, flush=True)
		print('Filename for image_id', image_id, ':', image_id_to_filename[image_id], flush=True)
		Null_pred_boxes.append(image_id)
		continue
	
	#max_box_s = result_bboxes[np.where(result_scores==np.max(result_scores))[0][0]]
	#max_keys_s = result_keypoints[np.where(result_scores==np.max(result_scores))[0][0]]
	#max_keypoints.append(max_keys_s)
	#max_boxes.append(max_box_s)
	#max_scores.append(np.max(result_scores))
	
	# Only choose the boxes that are in the central
	nboxes_in_center = 0
	res_box_cen_ids = []
	for res_i in range (len(result_bboxes)):
		if not bbox_in_center(result_bboxes[res_i], image_width, image_height):
			continue
		else:
			nboxes_in_center+=1
			res_box_cen_ids.append(res_i)

	#print('Number of central boxes for image id: ', image_id, ' are ', nboxes_in_center, flush=True)
	result_central_bboxes = [result_bboxes[i] for i in res_box_cen_ids]
	result_central_keypoints = [result_keypoints[i] for i in res_box_cen_ids]
	result_central_categories = [result_categories[i] for i in res_box_cen_ids]
	result_central_scores = [result_scores[i] for i in res_box_cen_ids]
	
	if not len(result_central_bboxes):
		#print('No Pred boxes in the centre for image_id:', image_id, flush=True)
		Null_pred_boxes_centre.append(image_id)
		result_central_bboxes = result_bboxes
		result_central_keypoints = result_keypoints
		result_central_categories = result_categories
		result_central_scores = result_scores

	max_box_s_c = result_central_bboxes[np.where(result_central_scores==np.max(result_central_scores))[0][0]]
	max_keys_s_c = result_central_keypoints[np.where(result_central_scores==np.max(result_central_scores))[0][0]]
	max_cats_s_c = result_central_categories[np.where(result_central_scores==np.max(result_central_scores))[0][0]]
	max_boxes_centre[image_id] = max_box_s_c  #.append(max_box_s_c)
	max_keypoints_centre[image_id] = max_keys_s_c   #.append(max_keys_s_c)
	max_categories_centre[image_id] = max_cats_s_c   #.append(max_cats_s_c)
	if calibrate_score:
		score_cal = frecal(float(np.max(result_central_scores)))
		if score_cal>1.:
			score_cal = 1.
		if score_cal<0.:
			score_cal = 0.
		max_scores_centre[image_id] = score_cal
	else:
		max_scores_centre[image_id] = np.max(result_central_scores)  #.append(np.max(result_central_scores))
	
	# Check if any box in all_boxes is inside central_box (excluding central_box itself)
	boxes_inside_central = []; keypoints_inside_central = []
	for idx_in, box in enumerate(result_bboxes):
		if box != max_box_s_c and is_box_inside(box, max_box_s_c):
			if result_scores[idx_in] >= score_cal:
				boxes_inside_central.append(box); keypoints_inside_central.append(result_keypoints[idx_in])
	small_box_in_max_boxes_centre[image_id] = boxes_inside_central or None
	small_keys_in_max_boxes_centre[image_id] = keypoints_inside_central or None

print('Number of images with no box prediction:', len(Null_pred_boxes), flush=True)
print('Number of images with no box prediction in centre:', len(Null_pred_boxes_centre), '(',round(len(Null_pred_boxes_centre)/len(image_ids)*100,1), '% )', flush=True)
print('Replaced', len(Null_pred_boxes_centre), 'boxes with no box prediction in centre with maximum score prediction anywhere in image.', flush=True)

# Filenames with no predictions
Null_filenames = [image_id_to_filename[i] for i in Null_pred_boxes]
Null_filenames_centre = [image_id_to_filename[i] for i in Null_pred_boxes_centre]

# Find out the number of times there is a smaller box inside max_boxes_centre
none_count = 0
# Iterate through the values in the dictionary
for value in small_box_in_max_boxes_centre.values():
    if value is None:
        none_count += 1
print('Fraction of sources with small box inside the bigger central box:', '(', round((len(image_ids)-none_count)/len(image_ids) * 100,1), '% )', flush=True)


####################################################################
########## Get the consolidated catalogue
####################################################################
# First sort with respect to the probabilities
max_scores_centre_sorted = {k: v for k, v in sorted(max_scores_centre.items(), key=lambda item: item[1], reverse=True)}
score25 = len({key: value for key, value in max_scores_centre_sorted.items() if value > 0.25})
print("Fraction of detections with probaility of detection > 0.25 =", score25/len(max_scores_centre_sorted), flush=True)

# Get a trustworthy score for FR-II
# Initialize an empty list to store the indexes
indexes_with_value_1 = []
# Iterate through the dictionary
for idfr2, catvalue in max_categories_centre.items():
    if catvalue == 1: # 1 is for FR-II
        indexes_with_value_1.append(idfr2)
scores_FR2 = [max_scores_centre_sorted[idfr2] for idfr2 in indexes_with_value_1]


# Get a trustworthy score for FR-X
# Initialize an empty list to store the indexes
indexes_with_value_3 = []
# Iterate through the dictionary
for idfrx, catvalue in max_categories_centre.items():
    if catvalue == 3: # 3 is for FR-X
        indexes_with_value_3.append(idfrx)
scores_FRx = [max_scores_centre_sorted[idfrx] for idfrx in indexes_with_value_3]


# Get a trustworthy score for R
# Initialize an empty list to store the indexes
indexes_with_value_4 = []
# Iterate through the dictionary
for idR, catvalue in max_categories_centre.items():
    if catvalue == 4: # 4 is for R
        indexes_with_value_4.append(idR)
scores_R = [max_scores_centre_sorted[idR] for idR in indexes_with_value_4]


#save the dictionary
with open(f"{out_dir}{SBid}/cons_catalogs/max_scores_centre_sorted{cat_version}.json", 'w') as json_file:
	json.dump(max_scores_centre_sorted, json_file)

# Make a copy of selavy catalog
tab = tab_selavy
positions = SkyCoord(np.array(tab['ra_deg_cont'])*u.deg, np.array(tab['dec_deg_cont'])*u.deg)
tab_consolidated = pd.DataFrame(columns=tab.columns)

tab_consolidated['ra_deg_cons'] = None
tab_consolidated['dec_deg_cons'] = None
tab_consolidated['flux_peak_cons'] = None
tab_consolidated['flux_int_cons'] = None
tab_consolidated['img_id_cons'] = None
tab_consolidated['filename_cons'] = None
tab_consolidated['radio_bbox_cons'] = None
tab_consolidated['PA'] = None
tab_consolidated['n_components'] = None
tab_consolidated['type'] = None
tab_consolidated['score'] = None
tab_consolidated['wise_ra_cons'] = None
tab_consolidated['wise_dec_cons'] = None

# Save the initial DataFrame to a CSV file
tab_consolidated.to_csv(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog{cat_version}.csv", index=False)

# Region files
if not os.path.exists(f"{out_dir}{SBid}/region_files"):
    os.mkdir(f"{out_dir}{SBid}/region_files")
# File path for the DS9 region file
output_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg"
# Open the DS9 region file for writing
with open(output_file, 'w') as f:
	# Write DS9 region file header
	f.write("# Region file format: DS9 CARTA 3.0.0\n")
	f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
	f.write("fk5\n")

	for _idx, key in enumerate(max_scores_centre_sorted, start=1):
	#for img_idx in range(len(image_ids)):
	
		# Choose an image ID
		image_id = key #image_ids[key]  
		#image_id = image_ids[img_idx]; _idx=0
	
		#if image_id != 4454: #17093:
		#	continue
	
		if image_id in Null_pred_boxes:
			continue
		else:
			print(image_id)
			
		# put a score limit on FR-II sources
		#if category_ids[max_categories_centre[image_id]] == 'FR-II':
		#	if max_scores_centre_sorted[image_id] <= np.percentile(scores_FR2, [50])[0]:
		#		continue
					
		# put a score limit on FR-X sources
		if category_ids[max_categories_centre[image_id]] != 'S':
			if max_scores_centre_sorted[image_id] <= 0.3:
				continue
	
		try:
			filename = image_id_to_filename[image_id]
			file = fits_file_dir + filename.replace('rgb.png', 'emu.fits') 
			dlist = fits.open(file)
			d = dlist[0]
			warnings.simplefilter('ignore')
			d_wcs = WCS(d.header).celestial
			d.data = np.squeeze(d.data)
			d_size_arcmin = d.data.shape[0]*img_pixel_size_arcsec/60.; nx, ny = d.data.shape
	
			################################
			## Consolidate radio
			################################
			#tab_idx = tab.iloc[list(tab['component_name'] == filename[:-8])]
			tab_idx = tab[tab.apply(lambda row: f"{row['ra_deg_cont']}_{row['dec_deg_cont']}" == filename[:-8], axis=1)]
			position = SkyCoord(np.array(tab_idx['ra_deg_cont'])*u.deg, np.array(tab_idx['dec_deg_cont'])*u.deg)
			matches = search_around_sky(position, positions, seplimit= d_size_arcmin*u.arcminute)
			src_matches = matches[1]
			spatial_pos = SKyToPixelTransformer(position, positions[src_matches], (0, 0.), pixel_scale=img_pixel_size_arcsec*u.arcsecond)
			source_x = np.array(spatial_pos.coords['offsets-pixel'][0].value + nx/2, dtype=int)
			source_y = np.array(spatial_pos.coords['offsets-pixel'][1].value + ny/2, dtype=int)
			#print('Number of Selavy catalogue sources in Image : ', len(source_x), flush=True)
	
			x_min, y_min, width, height = max_boxes_centre[image_id]
			x_center, y_center = x_min + width/2, y_min + height/2
			pmask_correction_applied = False
			if category_ids[max_categories_centre[image_id]] == 'S':
				reg = RectanglePixelRegion(PixCoord(x=x_center, y=y_center), width=width/2, height=height/2, angle=Angle(0, 'deg'))
			else:
				reg = RectanglePixelRegion(PixCoord(x=x_center, y=y_center), width=width, height=height, angle=Angle(0, 'deg'))
				if get_angles:
					srcs_in_box_init = reg.contains(PixCoord(x=source_x, y=source_y))
					srcs_in_box_ccord = PixCoord(x=source_x, y=source_y)[srcs_in_box_init]
					srcs_in_box_ccord_set = set(zip(srcs_in_box_ccord.x, srcs_in_box_ccord.y))
					noise = rms_estimate(d.data, mode='mad', clip_rounds=2)
					mask_img = np.zeros((d.data.shape))
					pmask = 50 # pixels to get rid of point sources in big box (median calculated on train set S sources)
					indv_img_masks = []
					for imask, mask in enumerate(pu.island_segmentation(d.data, 5.*noise)):
						if np.sum(mask) < pmask:
							continue
						argmask = np.argwhere(mask==True)
						argmaskp = PixCoord(argmask[:,1],argmask[:,0])
						argmaskp_set = set(zip(argmaskp.x, argmaskp.y))
						common_coords = srcs_in_box_ccord_set & argmaskp_set
						if any(reg.contains(argmaskp)) and common_coords:
							mask_img[mask] = 1
							indv_img_masks.append(np.sum(mask))
					contours,_ = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
					all_contours = np.concatenate(contours)
					(x_center,y_center),(width,height), angle = cv2.minAreaRect(all_contours)
					reg = RectanglePixelRegion(PixCoord(x=x_center, y=y_center), width=width, height=height, angle=Angle(angle, 'deg'))
					pmask_correction_applied = True
					if len(indv_img_masks)>1:
						mask_img = np.zeros((d.data.shape))
						pmask0 = np.sort(indv_img_masks)[::-1][0]; pmask1 = np.sort(indv_img_masks)[::-1][1]
						if pmask0/pmask1 > 2.: # Just a big enough difference in top and second-top number of pixels
							pmask = pmask0
						else:
							pmask = pmask1
						pmask_correction_applied = True
						print ('pmask correction applied for image_id', image_id, flush=True)
						for imask, mask in enumerate(pu.island_segmentation(d.data, 5.*noise)):
							if np.sum(mask) < pmask:
								continue
							argmask = np.argwhere(mask==True)
							argmaskp = PixCoord(argmask[:,1],argmask[:,0])
							argmaskp_set = set(zip(argmaskp.x, argmaskp.y))
							common_coords = srcs_in_box_ccord_set & argmaskp_set
							if any(reg.contains(argmaskp)) and common_coords:
								mask_img[mask] = 1
						contours,_ = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
						all_contours = np.concatenate(contours)
						(x_center,y_center),(width,height), angle = cv2.minAreaRect(all_contours)
						reg = RectanglePixelRegion(PixCoord(x=x_center, y=y_center), width=width, height=height, angle=Angle(angle, 'deg'))
				else:
					reg = RectanglePixelRegion(PixCoord(x=x_center, y=y_center), width=width, height=height, angle=Angle(0, 'deg'))
			srcs_in_box = reg.contains(PixCoord(x=source_x, y=source_y)) # True or False
			if not len(list(src_matches[srcs_in_box])):
				print ('----  for image_id', image_id, 'all sources in box already consolidated', flush=True)
				continue
			tab1 = tab.iloc[list(src_matches[srcs_in_box])]
			if category_ids[max_categories_centre[image_id]] != 'S':
				if len(tab1)==1 and category_ids[max_categories_centre[image_id]] != 'R':
					continue
			if len(tab1)>1:
				tab1_c = tab1.apply(to_list) # combining rows
			else:
				tab1_c = tab1

			# Get flux, not required actually can do it from selavy catalog
			"""peak_flux = []; int_flux = []; pmask = 5
			noise = rms_estimate(d.data.flatten(), mode='mad', clip_rounds=2)
			for iii in range(len(source_x[srcs_in_box])):
				for ii, mask in enumerate(island_segmentation(d.data, 3.*noise)):
					if np.sum(mask) <= pmask:
						continue
					argmask = np.argwhere(mask==True)
					if any(argmask[:,0] == source_y[iii]) and any(argmask[:,1] == source_x[iii]):
						peak_flux.append(np.max(d.data[mask])*1e3) #mJy
						int_flux.append(np.sum(d.data[mask])*1e3) #mJy"""
					
		
			# Fill values
			sky_position_r = PixelToSkyTransformer(position, (x_center, y_center), d_wcs)
			sky_coords_r = sky_position_r.coords["offsets-sky"]
			tab1_c['ra_deg_cons'] = sky_coords_r.ra.deg; tab1_c['dec_deg_cons'] = sky_coords_r.dec.deg
			tab1_c['flux_peak_cons'] = np.max(tab1['flux_peak'])
			tab1_c['flux_int_cons'] = np.sum(tab1['flux_int'])
			tab1_c['img_id_cons'] = image_id; tab1_c['filename_cons'] = filename[:-8]
			vals = [None] * len(tab1_c); vals[0] = [x_center, y_center, width, height, angle]
			tab1_c['radio_bbox_cons'] = vals
			tab1_c['PA'] = angle
			tab1_c['n_components'] = len(tab1)
			tab1_c['type'] = category_ids[max_categories_centre[image_id]]
			tab1_c['score'] = max_scores_centre_sorted[image_id]
			
			# region information
			f.write(f"box({(sky_coords_r.ra-5*u.arcsecond).deg}, {(sky_coords_r.dec+5*u.arcsecond).deg}, {width*img_pixel_size_arcsec}\", {height*img_pixel_size_arcsec}\", {angle}) # color=#2EE6D6 width=2 text={filename[:-8]}\n")
	
			"""
			# figure
			data_preprocess = radio_preprocess(d.data, pmask=50, lower=3, outer_pixels=True)
			import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle
			fig, ax = plt.subplots(1, 1)
			ax.imshow(data_preprocess)
			patch = reg.plot(ax=ax, facecolor='none', edgecolor='green', lw=4, label='Rectangle')
			ax.scatter(source_x[srcs_in_box], source_y[srcs_in_box], s=6, c='pink')
			plt.scatter(max_keypoints_centre[image_id][0], max_keypoints_centre[image_id][1], s=8, c='green')
			ax.invert_yaxis()
			plt.show()
			"""
	
			################################
			## Consolidate IR host
			################################
			x_pixels, y_pixels = np.array(max_keypoints_centre[image_id][0]), np.array(max_keypoints_centre[image_id][1])
			sky_position = PixelToSkyTransformer(position, (x_pixels, y_pixels), d_wcs)
			sky_coords = sky_position.coords["offsets-sky"]
			tab1_c['wise_ra_cons'] = sky_coords.ra.deg; tab1_c['wise_dec_cons'] = sky_coords.dec.deg
			if pmask_correction_applied:
				tab1_c['wise_ra_cons'] = tab1_c['ra_deg_cons']; tab1_c['wise_dec_cons'] = tab1_c['dec_deg_cons']
			if category_ids[max_categories_centre[image_id]] == 'S':
				tab1_c['wise_ra_cons'] = position.ra.deg[0]; tab1_c['wise_dec_cons'] = position.dec.deg[0]
			################################
			## Consolidated table
			################################
			tab = tab.drop(list(src_matches[srcs_in_box])) # drop used rows from main selavy dataframe
			tab = tab.reset_index(drop=True)
			positions = SkyCoord(np.array(tab['ra_deg_cont'])*u.deg, np.array(tab['dec_deg_cont'])*u.deg)
			tab_consolidated = pd.concat([tab_consolidated, tab1_c], ignore_index=True)
		
			# Incremental save
			tab1_c.to_csv(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog{cat_version}.csv", mode='a', header=False, index=False)
			print(_idx, image_id, len(tab1), len(tab_selavy), len(tab), len(tab_consolidated), position.separation(sky_coords).arcsecond[0], np.max(tab1['flux_peak']), np.sum(tab1['flux_peak']), category_ids[max_categories_centre[image_id]], filename[:-8], flush=True)
		except Exception as e:
			print(f"Error occurred in iteration: {e}", flush=True)
			continue

# Shuffle the dataframe to randomize compact and complex sources
#tab_consolidated = tab_consolidated.sample(frac=1).reset_index(drop=True)

# Also add unconsolidated leftover sources in tab
# First save it
tab.to_csv(f"{out_dir}{SBid}/cons_catalogs/catalog_leftover_selavy{cat_version}.csv", index=False)
tab['wise_ra_cons'] = tab['ra_deg_cont']; tab['wise_dec_cons'] = tab['dec_deg_cont']
tab['filename_cons'] = tab['component_name']
islands = Counter(tab['island_id'].values)
tab['n_components'] = tab.apply(lambda x: islands[x['island_id']], axis=1)
print(f"len(tab) and len(np.where(tab['n_components']>1)[0]) in left over selavy dataframe = {len(tab)}, {len(np.where(tab['n_components']>1)[0])}")
tab['type'] = 'S' # ATM only 52 out of 2075 in tab have n_components>1 in EMU-PS, come back to this later!
tab['score'] = 0
tab['flux_peak_cons'] = tab['flux_peak']; tab['flux_int_cons'] = tab['flux_int']
##### (tab does not have 'img_id_cons' or 'radio_bbox_cons' so can be separated later)
tab_consolidated = pd.concat([tab_consolidated, tab], ignore_index=True) # Also add unconsolidated leftover sources	in tab
tab_consolidated.to_csv(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last{cat_version}.csv", index=False)

# Write the table to a FITS file
data = pd.read_csv(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last{cat_version}.csv", usecols=['filename_cons', 'wise_ra_cons', 'wise_dec_cons', 'type', 'score'], low_memory=False)
data.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last{cat_version}.fits", format='fits', overwrite=True)

# Write FITS files and region files for all extended and different sources
data = pd.read_csv(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last{cat_version}.csv", usecols=['filename_cons', 'wise_ra_cons', 'wise_dec_cons', 'type', 'score', 'n_components'], low_memory=False)
#data.loc[(data['n_components'] == 1) & (data['type'] != 'R'), 'type'] = 'S'
#data.drop('n_components', axis=1, inplace=True)
# All extended with n_components>1
which_sources = 'multiSelavyInBbox'
data1 = data.iloc[list(data['n_components'] > 1)]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

# All extended
which_sources = 'allExtended'
data1 = data.iloc[list(data['type'] != 'S')]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

# FR-I sources
which_sources = 'FR-I'
data1 = data.iloc[list(data['type'] == which_sources)]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

# FR-II sources
which_sources = 'FR-II'
data1 = data.iloc[list(data['type'] == which_sources)]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

# FR-X sources
which_sources = 'FR-X'
data1 = data.iloc[list(data['type'] == which_sources)]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

# R sources
which_sources = 'R'
data1 = data.iloc[list(data['type'] == which_sources)]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

# C sources
which_sources = 'C'
data1 = data.iloc[list(data['type'] == which_sources)]; data1 = data1.reset_index(drop=True); data1.drop('n_components', axis=1, inplace=True)
data1.rename(columns={'wise_ra_cons': 'ra_cons', 'wise_dec_cons': 'dec_cons'}, inplace=True)
table = Table.from_pandas(data1)
table.write(f"{out_dir}{SBid}/cons_catalogs/consolidated_catalog_last_{which_sources}{cat_version}.fits", format='fits', overwrite=True)

source_filename = data1['filename_cons']
input_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_all{cat_version}.reg" 
output_reg_file = f"{out_dir}{SBid}/region_files/ds9_regions_{which_sources}{cat_version}.reg"
filter_ds9_regions(source_filename, input_reg_file, output_reg_file)

print ("Saved csv and fits catalog files and region files. Closing the log file now.", flush=True)
sys.stdout = old_stdout
log_file.close()		