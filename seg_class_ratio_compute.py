import os
from utils import *
from os.path import join
import numpy as np
import cv2
from sklearn.utils import shuffle



def load_imgs(data_paths, data_type, transform=True):
    # img_paths = list(img_paths)
    images = []

    for img_path in data_paths:
        img = cv2.imread(img_path)
        if data_type == 'mask':
            img = img/255
        
        if transform:
            # TODO padding and resize image to the same size
            norm_size = 256
            x, y, _ = img.shape
            if x > y:
                img = cv2.copyMakeBorder(img, 0, 0, 0, x-y, cv2.BORDER_CONSTANT, None, value = 0)
            if x < y:
                img = cv2.copyMakeBorder(img, 0, y-x, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
            assert img.shape[0] == img.shape[1]
            if img.shape[0] != norm_size:
                img = cv2.resize(img, (norm_size, norm_size))
        
        images.append(np.array(img))

    return np.array(images, dtype='uint8')


img_paths = []
mask_paths = []
for root, dirs, files in os.walk(SEG_DATA_PATH):
    if len(files) != 0:
        for file in files:
            if '.tif' in file: 
                img_paths.append(join(root, file))
            elif '.jpg' or '.gif' in file:
                mask_paths.append(join(root, file))

img_paths = sorted(img_paths)
mask_paths = sorted(mask_paths)
img_paths, mask_paths = shuffle(img_paths, mask_paths)
assert len(img_paths) == len(mask_paths), 'image and mask number don\'t match'

fg_points = 0
bg_points = 0
for mask_path in mask_paths:
    mask = cv2.imread(mask_path)[:,:,0]/255
    norm_size = 256
    x, y = mask.shape
    if x > y:
        mask = cv2.copyMakeBorder(mask, 0, 0, 0, x-y, cv2.BORDER_CONSTANT, None, value = 0)
    if x < y:
        mask = cv2.copyMakeBorder(mask, 0, y-x, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
    assert mask.shape[0] == mask.shape[1]
    if mask.shape[0] != norm_size:
        mask = cv2.resize(mask, (norm_size, norm_size))
    mask = np.array(mask, dtype=int)
    fg_point = np.sum(mask)
    bg_point = norm_size*norm_size-fg_point

    fg_points += fg_point
    bg_points += bg_point

print(bg_points/fg_points)


