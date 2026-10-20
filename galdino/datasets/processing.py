"""Some basic methods that can be employed for preprocessing of images to PINK. This is not meant to be an exhaustive set. Merely here as a reference. 
"""

import logging
from typing import Tuple, List, Dict, Iterator

import numpy as np
from skimage.morphology import closing, convex_hull_image
from skimage.measure import label
from astropy.convolution import convolve, Gaussian2DKernel

logger = logging.getLogger(__name__)

def minmax(data: np.ndarray) -> np.ndarray:
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

    mask = closing(data > threshold)

    img_labels, no_labels = label(mask, return_num=True)

    for i in range(no_labels):
        if i == 0 and return_background is False:
            continue

        island_mask = img_labels == i
        if np.sum(island_mask) <= minimum_island_size:
            continue
        yield island_mask


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


def radio_preprocess(data, pmask=2, lower=3, outer_pixels=True):
    nx, ny = data.shape
    noise = rms_estimate(data.flatten(), mode='mad', clip_rounds=2)
    if np.isnan(noise):
        noise = rms_estimate(data.flatten(), mode='std', clip_rounds=2)

    img_scale = np.zeros_like(data)
    for i, mask in enumerate(pu.island_segmentation(data, lower*noise)):
        if np.sum(mask) <= pmask:
            continue
        
        argmask = np.argwhere(mask==True)
        if any(argmask[:,0] == ny/2) and any(argmask[:,1] == nx/2):
            img_scale[mask] = pu.minmax(np.log10(data[mask]))        

        if outer_pixels:
            img_scale[mask] = pu.minmax(np.log10(data[mask]))
        
        img_scale[~np.isfinite(img_scale)] = 0   # to avoid nans in the image

    return img_scale.astype(np.float32)


def wise_preprocess(data, wpmask=2):
    nx, ny = data.shape
    med = np.median(data.flatten())
    mad = np.median(np.abs(data.flatten() - med))
    
    noise = med + 3*mad
    
    img_scale = np.zeros_like(data)
    for i, mask in enumerate(pu.island_segmentation(data, noise)):
        argmask = np.argwhere(mask==True)
        """for j in range(30):
            if any(argmask[:,0] == ny/2+j) and any(argmask[:,1] == nx/2):
                img_scale[mask] = pu.minmax(np.log10(data[mask]))
            if any(argmask[:,0] == ny/2-j) and any(argmask[:,1] == nx/2):
                img_scale[mask] = pu.minmax(np.log10(data[mask]))
            if any(argmask[:,0] == ny/2) and any(argmask[:,1] == nx/2+j):
                img_scale[mask] = pu.minmax(np.log10(data[mask]))
            if any(argmask[:,0] == ny/2) and any(argmask[:,1] == nx/2-j):
                img_scale[mask] = pu.minmax(np.log10(data[mask]))"""
        if np.sum(mask) <= wpmask:
            continue
        img_scale[mask] = pu.minmax(np.log10(data[mask]))
    
    return img_scale.astype(np.float32)


def preprocessing(cID, epmask=5, wpmask=5, lower=3, outer_pixels=True, wise=False, raw=False, complex_sources=True, inpath=None, reproject=True):
    #iID = cID.replace('component', 'island')[:-1]
    emu_src = f"{inpath}/{cID}_emu.fits" #f"{inpath}{iID}_{cID}_emu.fits"
    if wise:
        emu_src = f"{inpath[:-6]}/{cID}_emu.fits" #f"{inpath[:-6]}/{iID}_{cID}_emu.fits"
    #emu_src = f"../../Download_Images_Components/Images/{iID}_{cID}_emu.fits"
    
    try:
        dlist = fits.open(emu_src)
        d = dlist[0]
    except:
        return None
        
    d_wcs = WCS(d.header).celestial
    d.data = np.squeeze(d.data)
    d.header = d_wcs.to_header()
    
    if complex_sources:
        catalogs = tab_c
    else:
        catalogs = tab

    reproject_wcs = d.header.copy()
    reproject_wcs['CRVAL1'] = catalogs.loc[cID]['ra_deg_cont']
    reproject_wcs['CRVAL2'] = catalogs.loc[cID]['dec_deg_cont']
    reproject_wcs['CRPIX1'] = img_size[1] // 2
    reproject_wcs['CRPIX2'] = img_size[2] // 2
    reproject_wcs = WCS(reproject_wcs).celestial

    if reproject:
        e_new_data, e_new_footprint = reproject_interp(d, reproject_wcs, shape_out = img_size[1:])
    else:
        e_new_data = d.data
    
    if raw:
        e_new_data = e_new_data.astype(np.float32)
    else:
        e_new_data = radio_preprocess(e_new_data, epmask, lower, outer_pixels)
    dlist.close()
    
    e_new_data[~np.isfinite(e_new_data)] = 0.
    if np.sum(~np.isfinite(e_new_data)) > 0:
        return None
    if np.max(e_new_data) <= 0:
        print(e_new_data)
        #return None
    
    if wise is False:
        return (cID, np.array([e_new_data]))
    
    try:
        wlist = fits.open(f"{inpath}/{cID}_emu.fits".replace('emu','wise'))
        w = wlist[0]
    except:
        return None
    
    
    w_new_data, w_new_footprint = reproject_interp(w, reproject_wcs, shape_out = img_size[1:])
    #w_new_data = w.data
    
    if raw:
        w_new_data = w_new_data.astype(np.float32)
    else:
        w_new_data = wise_preprocess(w_new_data, wpmask)
    wlist.close()
    

    w_new_data[~np.isfinite(w_new_data)] = 0
    if np.sum(~np.isfinite(w_new_data)) > 0:
        return None
    if np.max(w_new_data) <= 0:
        return None
    
    return (cID, np.array((e_new_data, w_new_data)))

