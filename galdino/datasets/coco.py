# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from PIL import Image
import os.path
from astropy.io import fits
import numpy as np
import code
import cv2

import datasets.transforms as T

class CocoDetection_dataset(torchvision.datasets.vision.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        noisy_data,
        noisy_datax2,
        transform = None,
        target_transform = None,
        transforms = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
        
    def _load_image_fits(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"].replace('rgb.png', 'emu.fits')
        r = fits.open(os.path.join(self.root, path))[0].data
        #r  = r0[~np.isfinite(r0)] = 0.
        path = self.coco.loadImgs(id)[0]["file_name"].replace('emu.fits', 'rgb.png')
        im = np.array(Image.open(os.path.join(self.root, path).replace('_fits', '2017')).convert("RGB"))
        maxclips = [95, 99, 99.2, 99.5, 99.7, 99.9, 99.99]
        img = np.zeros((r.shape[0], r.shape[1], len(maxclips)+1), dtype=np.float32)
        #img[:,:,1] = im[:,:,2] # Radio preprocessed
        img[:,:,0] = im[:,:,0] # Infrared preprocessed
        for clipid, maxclip in enumerate(maxclips):
        	dataclips = np.clip(r, np.nanpercentile(r, [50]),np.nanpercentile(r, [maxclip]))
        	img[:,:,len(maxclips)-clipid] = dataclips #/np.nanmedian(dataclips)
        return img
        
    def _load_image_fitsx2(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"].replace('rgb.png', 'emu.fits')
        r = fits.open(os.path.join(self.root, path))[0].data
        path = self.coco.loadImgs(id)[0]["file_name"].replace('emu.fits', 'rgb.png')
        im = np.array(Image.open(os.path.join(self.root, path).replace('_fits', '2017')).convert("RGB"))
        if r.shape != im[:,:,0].shape or np.isnan(r).any():
            return Image.fromarray(im, "RGB")
        r = ((2**32-1)*r).astype(np.uint32)
        img = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
        #img[:,:,2] = im[:,:,2]; img[:,:,1] = r # Radio preprocessed and raw
        img[:,:,2] = r >> 24; img[:,:,1] = (r >> 16) & 0xFF # Radio raw and raw
        img[:,:,0] = im[:,:,0] # Infrared preprocessed
        return Image.fromarray(img, "RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        if self.noisy_data:
            image = self._load_image_fits(id)
        if self.noisy_datax2:
            image = self._load_image_fitsx2(id)
        if self.noisy_data ==False and self.noisy_datax2==False:
            image = self._load_image(id)
        
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(CocoDetection_dataset):
    def __init__(self, img_folder, ann_file, noisy_data, noisy_datax2, bbox_rotated, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file, noisy_data, noisy_datax2)
        self._transforms = transforms
        self.noisy_data = noisy_data
        self.noisy_datax2 = noisy_datax2
        self.bbox_rotated = bbox_rotated
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img_, target_ = self.prepare(img, target)      
        if self._transforms is not None:
            while True:
            	img, target = self._transforms(img_, target_)
            	if (not self.bbox_rotated or len(target['bbox_rotated']) > 0) and len(target['boxes']) > 0:
            		break
            	print("Re-cropped and resized.")
        if self.bbox_rotated:
        	target['boxes'] = target['bbox_rotated']
        return img, target

class CocoDetection_all(CocoDetection_dataset):
    def __init__(self, img_folder, ann_file, noisy_data, noisy_datax2, bbox_rotated, transforms, return_masks):
        super(CocoDetection_all, self).__init__(img_folder, ann_file, noisy_data, noisy_datax2)
        self._transforms = transforms
        self.noisy_data = noisy_data
        self.noisy_datax2 = noisy_datax2
        self.bbox_rotated = bbox_rotated
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection_all, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img_, target_ = self.prepare(img, target)      
        if self._transforms is not None:
            img, target = self._transforms(img_, target_)
        return img, target

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[0], image.shape[1]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            """if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)"""

        bbox_rotated = None
        if anno and "bbox_rotated" in anno[0]:
            bbox_rotated = [obj["bbox_rotated"] for obj in anno]
            bbox_rotated = torch.as_tensor(bbox_rotated, dtype=torch.float32).reshape(-1, 4)
            #bbox_rotated[:, 2:] += bbox_rotated[:, :2]
            bbox_rotated[:, 0::2].clamp_(min=0, max=w)
            bbox_rotated[:, 1::2].clamp_(min=0, max=h)
        
        angle = None
        if anno and "angle" in anno[0]:
            angle = [obj["angle"] for obj in anno]
            angle = torch.as_tensor(angle, dtype=torch.float32)
        
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if bbox_rotated is not None:
            bbox_rotated = bbox_rotated[keep]
        if angle is not None:
            angle = angle[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        if bbox_rotated is not None:
            target["bbox_rotated"] = bbox_rotated
        if angle is not None:
            target["angle"] = angle

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args, crop=True):
    
    if args.noisy_data:
        import datasets.transforms as T
        normalize = T.Compose([
            #T.ToTensor(),
            T.Normalize([17.713925925925928, 2.89831549425502e-05, 
                         2.4469861619116223e-05, 2.0068872320164804e-05, 1.7966315339084384e-05, 
                         1.6244567390773507e-05, 1.557404843904875e-05, 1.2022589927246241e-05], 
                        [47.795902394418725, 0.0002943887363691984, 
                         0.00015363376918687703, 7.412588806984124e-05, 5.20151836779506e-05, 
                         3.7978901303466134e-05, 3.354333050116355e-05, 1.9338942347847836e-05])
            ])
            
    if args.noisy_datax2:
        import datasets.transformsPNG as T
        normalize = T.Compose([
            T.ToTensor(),
            #T.Normalize([0.06935824, 0.016939376, 0.016241161], [0.18748689, 0.10635011, 0.10272085])
            T.Normalize([0.06935824, 0.3000091, 0.29598746], [0.18748689, 0.4532347, 0.4564638]) #Raw radio
            ])
    
    if args.noisy_data==False and args.noisy_datax2==False:
        import datasets.transformsPNG as T
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.06935824, 0.016939376, 0.016241161], [0.18748689, 0.10635011, 0.10272085]) # preprocessed radio
            #T.Normalize([0.06935824, 0.3000091, 0.29598746], [0.18748689, 0.4532347, 0.4564638])
            ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #scales = [224, 224+32, 224+2*32, 224+3*32, 224+4*32, 224+5*32, 224+6*32, 224+7*32]
    max_size = 1333 #800 #448

    if args.noisy_data:
        if image_set == 'train':
    	    if crop:
    		    return T.Compose([
    		        T.ToTensor(),
    			    T.RandomHorizontalFlip(),
    			    T.RandomRotation(),
    			    T.RandomSelect(
    			    T.RandomResize(scales, max_size=max_size),
    				    T.Compose([
    					    T.RandomResize([400, 500, 600]), #([224, 336, 448]),
    					    T.RandomSizeCrop(384, 384), #(128, 128), 
    					    T.RandomResize(scales, max_size=max_size),
    				    ])
    			    ),
    			    normalize,
    			    ])
    	    else:
    		    return T.Compose([
    		        T.ToTensor(),
    			    T.RandomHorizontalFlip(),
    			    T.RandomRotation(),
    			    T.RandomResize(scales, max_size=max_size),
    			    normalize,
    		    ])

        if image_set == 'val' or image_set == 'test' or image_set == 'all':
            return T.Compose([
                T.ToTensor(),
                T.RandomResize([800], max_size=max_size), #256
                normalize,
            ])

    else:
        if image_set == 'train':
    	    if crop:
    		    return T.Compose([
    		        #T.ToTensor(),
    			    T.RandomHorizontalFlip(),
    			    T.RandomRotation(),
    			    T.RandomSelect(
    			    T.RandomResize(scales, max_size=max_size),
    				    T.Compose([
    					    T.RandomResize([400, 500, 600]), #([224, 336, 448]),
    					    T.RandomSizeCrop(384, 384), #(128, 128), 
    					    T.RandomResize(scales, max_size=max_size),
    				    ])
    			    ),
    			    normalize,
    			    ])
    	    else:
    		    return T.Compose([
    		        #T.ToTensor(),
    			    T.RandomHorizontalFlip(),
    			    T.RandomRotation(),
    			    T.RandomResize(scales, max_size=max_size),
    			    normalize,
    		    ])

        if image_set == 'val' or image_set == 'test' or image_set == 'all':
            return T.Compose([
                #T.ToTensor(),
                T.RandomResize([800], max_size=max_size), #256
                normalize,
            ])
     
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances_angle'
    """if args.instances_random5percent:
        mode1 = 'instances_random10percent' 
    if args.instances_random5percent:
        mode1 = 'instances_random10percent'
    if args.instances_random10percent_seed1:
        mode1 = 'instances_random10percent_seed1' 
    if args.instances_random10percent_seed2:
        mode1 = 'instances_random10percent_seed2' 
    if args.instances_random10percent_seed3:
        mode1 = 'instances_random10percent_seed3'
    if args.instances_random10percent_seed4:
        mode1 = 'instances_random10percent_seed4'
    if args.instances_random10percent_seed5:
        mode1 = 'instances_random10percent_seed5'
    if args.instances_random10percent_ext:
        mode1 = 'instances_random10percent_ext'
    else:"""
    mode1 = mode
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode1}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'{mode}_test2017.json'),
    }
    if args.noisy_data or args.noisy_datax2:
        PATHS = {
            "train": (root / "train_fits", root / "annotations" / f'{mode1}_train2017.json'),
            "val": (root / "val_fits", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root / "test_fits", root / "annotations" / f'{mode}_test2017.json'),
        }
    if args.noisy_data and (args.data_eval=='all'):
        PATHS = {
            "train": (root / "train_fits", root / "annotations" / f'{mode1}_train2017.json'),
            "val": (root / "val_fits", root / "annotations" / f'{mode}_val2017.json'),
            "all": (root / f"all_SBs/{args.SBid}/All_fits", root / f"all_SBs/{args.SBid}" / f'{mode}_{args.SBid}.json'),
        }

    img_folder, ann_file = PATHS[image_set]
    if args.data_eval=='all':
        dataset = CocoDetection_all(img_folder, ann_file, args.noisy_data, args.noisy_datax2, args.bbox_rotated, transforms=make_coco_transforms(image_set, args), return_masks=True)
    else:
        dataset = CocoDetection(img_folder, ann_file, args.noisy_data, args.noisy_datax2, args.bbox_rotated, transforms=make_coco_transforms(image_set, args), return_masks=True)
    return dataset


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from util.box_ops import box_cxcywh_to_xyxy

    class DefaultArgs():
        def __init__(self):
            self.coco_path = 'CatalogNET_V1'
            self.masks = None
            self.keypoints = True
            self.noisy_data = True
            self.noisy_datax2 = False
            self.data_eval = 'train'
            self.bbox_rotated = True

    args = DefaultArgs()
    ds = build(image_set='train', args=args)
    
    get_mean_std = False  # This is done only after T.ToTensor() (commenting other tranformations)
    means = []; stds = []
    

    for i in range(3004,3005):
        img, target = ds[i]
        #print(img.shape, img.shape[-2:], target['image_id'])
        #print(target['keypoints'] * torch.tensor([iw, ih, 1], dtype=torch.float32))
        if get_mean_std:
            mean, std = img.mean([1,2]), img.std([1,2])
            means.append(mean.numpy()); stds.append(std.numpy())
            #print (means, stds)
            continue
        ih, iw = img.shape[-2:]
        #print(target["boxes"])
        target["boxes"] = target["boxes"] * torch.tensor([iw, ih, iw, ih], dtype=torch.float32)
        target["bbox_rotated"] = target["bbox_rotated"] * torch.tensor([iw, ih, iw, ih], dtype=torch.float32)
        #print(target["boxes"])
        img = img.permute((1,2,0))
        plt.imshow(img[:,:,6]) #plt.imshow(img)
        """fig = plt.figure(figsize=(15,10))
        plt.subplot(3, 3, 1); plt.imshow(img[:,:,0], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 2); plt.imshow(img[:,:,1], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 3); plt.imshow(img[:,:,2], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 4); plt.imshow(img[:,:,3], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 5); plt.imshow(img[:,:,4], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 6); plt.imshow(img[:,:,5], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 7); plt.imshow(img[:,:,6], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')
        plt.subplot(3, 3, 8); plt.imshow(img[:,:,7], cmap='hot', interpolation='none');plt.colorbar(); plt.title('')"""
        plt.axis('off')
        ax = plt.gca()
        ax.set_autoscale_on(False)
        print("Total boxes: {}".format(len(target["boxes"])))
        if args.bbox_rotated:
            angles = target['angle'].numpy()* 180.
            print('Angles: ', angles)
            #print('Rotated boxes: ', target["bbox_rotated"])
            #for idx, ann in enumerate(target['bbox_rotated'].numpy()):
            print('Rotated boxes: ', target["boxes"])
            for idx, ann in enumerate(target['boxes'].numpy()):
                angle = angles[idx]
                [x, y, w, h] = ann
                rect = ((x, y), (w, h), angle)
                box = cv2.boxPoints(rect) 
                box = np.intp(box)
                rectangle = plt.Polygon(box, closed=True, edgecolor='r', fill=None, linewidth=2)
                ax.add_patch(rectangle)
                ax.text(x+w/2, y+h/2, "{}".format(target['labels'][idx]), color="red", size=14, backgroundcolor="black")
        else:
            print(target["boxes"])
            target["boxes"] = box_cxcywh_to_xyxy(target["boxes"])
            for idx, ann in enumerate(target['boxes']):
                [x1, y1, x2, y2] = ann
                p = Rectangle((x1, y1), (x2-x1), (y2-y1), linewidth=2,
                                alpha=0.7, linestyle="solid",
                                edgecolor="red", facecolor='none')
                ax.add_patch(p)
                ax.text(x, y, "{}".format(target['labels'][idx]), color="red", size=14, backgroundcolor="black")
        for idx, ann_keys in enumerate(target['keypoints']):
            x, y, zz = ann_keys * torch.tensor([iw, ih, 1], dtype=torch.float32)
            print('Keypoints: ', ann_keys * torch.tensor([iw, ih, 1], dtype=torch.float32))
            ax.scatter(x, y, s =50, color='orange')
        plt.show()
    
    if get_mean_std:
        print (np.median(np.array(means)[:,0]), np.median(np.array(means)[:,1]), np.median(np.array(means)[:,2]),
               np.median(np.array(means)[:,3]), np.median(np.array(means)[:,4]), np.median(np.array(means)[:,5]),
               np.median(np.array(means)[:,6]), np.median(np.array(means)[:,7]) ) 
        print (np.median(np.array(stds)[:,0]), np.median(np.array(stds)[:,1]), np.median(np.array(stds)[:,2]), 
               np.median(np.array(stds)[:,3]), np.median(np.array(stds)[:,4]), np.median(np.array(stds)[:,5]),
               np.median(np.array(stds)[:,6]), np.median(np.array(stds)[:,7]) )

    