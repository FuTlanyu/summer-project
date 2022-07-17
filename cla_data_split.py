'''
Rearrange the train / validation / test set (8/1/1)
'''

import os
import numpy as np
import cv2
import random
import shutil
from os.path import join
import matplotlib.pyplot as plt
from utils import CLA_DATA_OLD_PATH, CLA_DATA_NEW_PATH

TRAIN_DATA_PATH = join(CLA_DATA_OLD_PATH, 'train')
TEST_DATA_PATH = join(CLA_DATA_OLD_PATH, 'test')
VAL_DATA_PATH = join(CLA_DATA_OLD_PATH, 'val')

TRAIN_AF_PATH = join(TRAIN_DATA_PATH, 'AF')
TRAIN_SLO_PATH = join(TRAIN_DATA_PATH, 'SLO')

TEST_AF_PATH = join(TEST_DATA_PATH, 'AF')
TEST_SLO_PATH = join(TEST_DATA_PATH, 'SLO')

VAL_AF_PATH = join(VAL_DATA_PATH, 'AF')
VAL_SLO_PATH = join(VAL_DATA_PATH, 'SLO')

train_af = os.listdir(TRAIN_AF_PATH)
train_slo = os.listdir(TRAIN_SLO_PATH)

test_af = os.listdir(TEST_AF_PATH)
test_slo = os.listdir(TEST_SLO_PATH)

val_af = os.listdir(VAL_AF_PATH)
val_slo = os.listdir(VAL_SLO_PATH)

af = train_af + test_af + val_af  # 514
slo = train_slo + test_slo + val_slo  # 1038

dict_all = {}
for img in af:
    dict_all[img] = 1

for img in slo:
    dict_all[img] = 0



all_files = []
all_file_names = []
for root, dirs, files in os.walk(CLA_DATA_OLD_PATH):
    if len(files) != 0:
        for file in files:
            all_files.append(os.path.join(root, file))
            all_file_names.append(file)

'''
Check if there exists some duplicate data in the original dataset
20-39 / 934-953 
'''
all_file_names_iden = list(set(all_file_names))
dup_items = []
for id_filename in all_file_names:
    idx = [i for i, e in enumerate(all_file_names) if e == id_filename]
    if len(idx) != 1:
        # dup_items.append([all_files[i] for i in idx])
        dup_items.append(idx)
print(dup_items)
print('20: ', all_files[20])
print('20: ', all_file_names[20])
print('934: ', all_file_names[934])
print('39: ', all_file_names[39])
print('953: ', all_file_names[953])

# display duplicate items
num = 20
fig, axs = plt.subplots(2, num)
for i in range(num):
    img = cv2.imread(all_files[i+20])
    axs[0, i].imshow(img)
    axs[0, i].axis('off')

for i in range(num):
    img = cv2.imread(all_files[i+934])
    axs[1, i].imshow(img)
    axs[1, i].axis('off')
plt.savefig('dup_display.png')


# remove duplicate files
for i in range(20):
    all_files.remove(all_files[i+934])


# absolute path and label
dict_all = {}
af = 0
slo = 0
for file in all_files:
    path_list = os.path.normpath(file).split(os.path.sep)
    if 'AF' in path_list:
        dict_all[file] = 1
        af += 1
    if 'SLO' in path_list:
        dict_all[file] = 0
        slo += 1



# shuffle the dict then split (8/1/1)
keys = list(dict_all.keys())    
random.shuffle(keys)
dict_random = [(key, dict_all[key]) for key in keys]

num = len(dict_random)
s1 = int(0.8*num)
s2 = int(0.9*num)
dict_train = dict_random[:s1]
dict_test = dict_random[s1:s2]
dict_val = dict_random[s2:]

for file in dict_train:
    src = file[0]
    label = file[1]
    if label == 1:
        dir = os.path.join(CLA_DATA_NEW_PATH, 'train', 'AF')
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dir)
    else:
        dir = os.path.join(CLA_DATA_NEW_PATH, 'train', 'SLO')
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dir)

for file in dict_test:
    src = file[0]
    label = file[1]
    if label == 1:
        dir = os.path.join(CLA_DATA_NEW_PATH, 'test', 'AF')
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dir)
    else:
        dir = os.path.join(CLA_DATA_NEW_PATH, 'test', 'SLO')
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dir)


for file in dict_val:
    src = file[0]
    label = file[1]
    if label == 1:
        dir = os.path.join(CLA_DATA_NEW_PATH, 'val', 'AF')
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dir)
    else:
        dir = os.path.join(CLA_DATA_NEW_PATH, 'val', 'SLO')
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dir)


