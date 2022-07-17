from utils import *
import os
import cv2
import numpy as np
from PIL import Image


# transfer a gif mask to jpg file
# gif 1:black 0:white
# file_path = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\u_gif\1912044_OD_2015.gif'
# mask = np.array(Image.open(file_path))*255
# mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
# im = Image.fromarray(mask)
# im.save(r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\u_gif\1912044_OD_2015.jpg')


# inspect mask image attribute
mask = cv2.imread(r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\data_comb\221131_OD_2013.jpg')
mask = np.array(mask)

assert np.array_equal(mask[:,:,0], mask[:,:,1]), 'not equal (0,1)' 
assert np.array_equal(mask[:,:,1], mask[:,:,2]), 'not equal (1,2)' 
xx = mask[:,:,0]/255
print('max: ', np.max(xx))
print('min: ', np.min(xx))


img_paths = []
mask_paths = []
for root, dirs, files in os.walk(SEG_DATA_PATH):
    if len(files) != 0:
        for file in files:
            if '.tif' in file: 
                img_paths.append(os.path.join(root, file))
            elif '.jpg' or '.gif' in file:
                mask_paths.append(os.path.join(root, file))
                if '.gif' in file:
                    xx = Image.open(os.path.join(root, file))
                    xx = np.array(xx)
                    print(os.path.join(root, file))

for i in range(len(img_paths)):
    img_name = os.path.split(img_paths[i])[-1].split('.')[0]
    mask_name = os.path.split(mask_paths[i])[-1].split('.')[0]
    assert img_name == mask_name, "image and mask don't match"


# compute image size distribution
img_size_dict = {}
for img_path in img_paths:
    img = cv2.imread(img_path) 
    shape = img.shape
    keys = list(img_size_dict.keys())
    if shape not in keys:
        img_size_dict[shape] = 1
    else:
        img_size_dict[shape] += 1

print(img_size_dict)






print('finished')