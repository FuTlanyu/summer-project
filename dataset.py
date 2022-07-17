import os
import random

from utils import *
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from torch.utils.data import Dataset
import h5py


'''
Dataset for segmentation
'''
class AFDataset(Dataset):
    def __init__(self, type, augmentation = False):
        # open the h5 file
        assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'
        self.type = type
        h5_file_name = join(SEG_H5_FILE_PATH, f'{self.type}.h5')
        self.hdf5 = h5py.File(h5_file_name, 'r')
        self.augmentation = augmentation
        if self.type != 'train':
            self.file_keys = list(self.hdf5.keys())
            # filenames = [self.file_keys[i] for i in range(0, len(self.file_keys), 2)]
            # print(f'{self.type} dataset: {filenames}')

    def __len__(self):
        if self.type == 'train':
            self.length = self.hdf5['masks'].shape[0]
        else: 
            self.length = int(len(self.file_keys)/2)
        
        return self.length
    
    def __getitem__(self, index):
        filename = ''
        mask_origin = np.zeros(1)
        if self.type == 'train':        
            img = self.hdf5['imgs'][index]
            mask = self.hdf5['masks'][index]
            
            if self.augmentation:
                img, mask = self.augment(img, mask) 

        else:
            filename = self.file_keys[2*index]
            img = self.hdf5[filename][0]
            mask = self.hdf5[filename][1]
            mask_origin = self.hdf5[filename+'_origin'][:,:,-1]
        
        # transform to tensor
        img = TF.to_tensor(img)
        mask = torch.tensor(np.array(mask)[:,:,-1], dtype=torch.long)

        normalise = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = normalise(img)

        return img, mask, mask_origin, filename

    def augment(self, img, mask):
        # transform to pillow image
        img = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)

        # # random rotation
        # deg = T.RandomRotation.get_params((0,180))
        # img = TF.rotate(img, deg)
        # mask = TF.rotate(mask, deg)

        # random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # random vertical flipping
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        return img, mask

    def close(self):
        self.hdf5.close()




'''
Dataset for classification
'''
class ASDataset(Dataset):
    def __init__(self, type, transform):
        # open the h5 file
        assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'
        h5_file_name = join(CLA_H5_FILE_PATH, f'{type}.h5')
        self.dset = h5py.File(h5_file_name, 'r')
        self.length = self.dset['labels'].shape[0]
        
        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
                
        img = self.dset['imgs'][index]
        # (c, 3, 256, 256)
        img = self.transform(img)
        label = self.dset['labels'][index]        
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def close(self):
        self.dset.close()



'''
Dataset for multi-task learning
'''
