# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_utils.ipynb.

# %% auto 0
__all__ = ['load_obj_model', 'detect_objects', 'centroid', 'list_centroids', 'inter_dist', 'show_mask', 'show_points', 'show_box',
           'show_anns', 'get_mask_area', 'calculateIoU', 'segment_with_prompts', 'load_sam_model', 'segment_everything',
           'segment']

# %% ../nbs/00_utils.ipynb 2
from .imports import *
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import skimage.transform as st
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam

# %% ../nbs/00_utils.ipynb 4
def load_obj_model(name="yolov8n.pt"):
    return YOLO(name)


def detect_objects(model, img):
    res = model(img, stream=True)
    return [{"boxes": r.boxes.data.detach().cpu().tolist()} for r in res]


def centroid(l):
    t = []
    cx = (l[0] + l[2]) / 2.0
    cy = (l[1] + l[3]) / 2.0
    t.append(cx)
    t.append(cy)
    return t


def list_centroids(objects):
    c = []
    for i in range(0, 11):
        l = []
        for j in range(0, 4):
            l.append(objects[0].get("boxes")[i][j])
        centre = centroid(l)
        c.append(centre)
    return c


def inter_dist(objects):
    c = list_centroids(objects)
    dis = []
    st = []
    for i in range(0, 11):
        for j in range(i + 1, 11):
            # st.append("Distance b/w object "+str(i)+" and object "+str(j))
            # st.append("D("+str(i)+","+str(j)+")")
            dis.append(math.dist(c[i], c[j]))
    # return st,dis
    return dis

# %% ../nbs/00_utils.ipynb 5
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # print(mask_image.shape)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


# %% ../nbs/00_utils.ipynb 6
def get_mask_area(mask:np.ndarray):
  area = mask.sum() # assumes binary mask (True == 1)
  return area

def calculateIoU(gtMask, predMask):
        # Calculate the true positives,
        # false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0

        for i in range(gtMask.shape[0]):
            for j in range(gtMask.shape[1]):
                if gtMask[i][j] == 1 and predMask[i][j] == 1:
                    tp += 1
                elif gtMask[i][j] == 0 and predMask[i][j] == 1:
                    fp += 1
                elif gtMask[i][j] == 1 and predMask[i][j] == 0:
                    fn += 1
        # Calculate IoU
        iou = tp / (tp + fp + fn)

        return iou

def segment_with_prompts(sam_model:Sam, image:np.ndarray, **kwargs):
  h,w,_ = image.shape
  points=np.array([[w*0.5, h*0.5], [0, h], [w, 0], [0,0], [w,h]])
  labels = np.array([1, 0, 0, 0, 0])
  mask = kwargs.get('mask', None)
  mask = st.resize(mask, (256, 256), order=0, preserve_range=True, anti_aliasing=False)
  mask = np.stack((mask,)*1, axis = 0)
  predictor = SamPredictor(sam_model)
  predictor.set_image(image)
  masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels, mask_input=mask, multimask_output=False)
  return masks

def load_sam_model(sam_checkpoint:str = "sam_vit_h_4b8939.pth", model_type:str = "vit_h", device:str = "cuda"):
  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)
  return sam

def segment_everything(sam_model:Sam, image:np.ndarray, **kwargs):
  mask = kwargs['mask']
  mask_generator = SamAutomaticMaskGenerator(sam_model)
  masks = mask_generator.generate(image)
  sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
  best = -1.0
  ind = -100
  area1 = get_mask_area(mask.astype(int))
  for i in range(10):
    val = calculateIoU(mask.astype(int), sorted_anns[i]['segmentation'].astype(int))
    area2 = get_mask_area(sorted_anns[i]['segmentation'].astype(int))
    dif = abs(area2 - area1)
    if val > best and dif < 5000:
      ind = i
      best = val
    elif val > best:
      ind = i
      best = val
  return masks[ind]

def segment(sam_model:Sam, image:np.ndarray, seg_function=segment_with_prompts, **kwargs):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  mask_fname = kwargs.get('mask_path', 'mask.png')
  mask = cv2.imread(mask_fname)
  mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
  mask = mask.astype(bool)
  masks = seg_function(sam_model, image, mask=mask)
  return masks
