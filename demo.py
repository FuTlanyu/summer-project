import os
import random

from utils import *

from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch import optim
from torch.utils.data import IterableDataset, Dataset, DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import KFold
import h5py
import time
from itertools import cycle, islice

# data = np.arange(10)
# N_FOLD = 3
# kf = KFold(n_splits=N_FOLD)

# # test 
# kf_gene = kf.split(data)
# for i in range(N_FOLD):
#     data = next(kf_gene)
#     print(data)
# print('finished')


# iterable dataset try out
# class MyIter(IterableDataset):
#     def __init__(self, data):
#         self.data = data

#     def process_data(self, data):
#         for x in data:
#             yield x
    
#     def __iter__(self):
#         return cycle(self.process_data(self.data))


# data = np.arange(12)
# iter_dset = MyIter(data)
# loader = DataLoader(iter_dset, batch_size=5, shuffle=False)

# for batch in islice(loader, 10):
#     print(batch)

class MyDset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data = np.arange(12)
my_dd = MyDset(data)
trainloader = DataLoader(my_dd, batch_size=5, shuffle=True, drop_last=True)

train_iter = iter(trainloader)
for i in range(5):
    for j in range(1):
        try:
            data = next(train_iter)
            print(data)
        except StopIteration:
            train_iter = iter(trainloader)
            data = next(train_iter)
            print(data)
            # if begin_data is not None:
            #     if torch.equal(begin_data, data[0]):
            #         print('Same')
            #     else:
            #         print('Diff')
            
            # begin_data = data[0]




# import gc
# for obj in gc.get_objects():   # Browse through ALL objects
#     if isinstance(obj, h5py.File):   # Just HDF5 files
#         try:
#             obj.close()
#         except:
#             pass # Was already closed

# save_best_model = True
# test_ratio = 0.2
# N_FOLD = 5
# BATCHSIZE_TR = 4
# BATCHSIZE_TE = 1
# LR = 0.001
# EPOCH = 1    
# aug_train_bool = False

# bayes_bool = False
# dropout = 0.2
# uncertain_ana = False 
# num_runs_bayes = 20

# loss_type = 'dicebce'


# print('Loading data...')
# img_paths = []
# mask_paths = []
# for root, dirs, files in os.walk(SEG_DATA_PATH):
#     if len(files) != 0:
#         for file in files:
#             if '.tif' in file: 
#                 img_paths.append(os.path.join(root, file))
#             elif '.jpg' or '.gif' in file:
#                 mask_paths.append(os.path.join(root, file))

# img_paths = sorted(img_paths)
# mask_paths = sorted(mask_paths)
# img_paths, mask_paths = shuffle(img_paths, mask_paths)
# assert len(img_paths) == len(mask_paths), 'image and mask number don\'t match'
# N = len(img_paths)

# for i in range(len(img_paths)):
#     img_name = os.path.split(img_paths[i])[-1].split('.')[0]
#     mask_name = os.path.split(mask_paths[i])[-1].split('.')[0]
#     assert img_name == mask_name, "image and mask don't match"

# ### Split dataset into training and test set
# split_idx = int(N*test_ratio)
# test_imgs = img_paths[:split_idx]
# test_masks = mask_paths[:split_idx]
# print(f'test size {len(test_imgs)}')

# create_h5_test_seg(test_imgs, test_masks, 'test')

# trv_imgs = img_paths[split_idx:]
# trv_masks = mask_paths[split_idx:]

# # miou of the last epoch
# val_miou_ds_per_fold = np.zeros(N_FOLD)
# val_miou_per_fold = np.zeros(N_FOLD)
# train_miou_per_fold = np.zeros(N_FOLD)
# # miou of the best epoch
# val_miou_best_per_fold = np.zeros(N_FOLD)

# kf = KFold(n_splits=N_FOLD)
# for k_i, (train_index, val_index) in enumerate(kf.split(trv_imgs)):

#     print(f'\n{k_i+1} fold model')
#     print(f'{k_i+1} fold train size: {len(train_index)} val size: {len(val_index)}')
#     train_imgs = []
#     train_masks = []
#     for idx in train_index:
#         train_imgs.append(trv_imgs[idx])
#         train_masks.append(trv_masks[idx])
    
#     val_imgs = []
#     val_masks = []
#     for idx in val_index:
#         val_imgs.append(trv_imgs[idx])
#         val_masks.append(trv_masks[idx])


#     # create hdf5 file for each split
#     print('Generating hdf5 file...')
#     t1 = time.time()
#     create_h5_train_seg(train_imgs, train_masks, 'train')
#     create_h5_test_seg(val_imgs, val_masks, 'val')
#     t2 = time.time()
#     print('hdf5 file generation time: %.2f (min)' % ((t2-t1)/60))

#     trainset = AFDataset('train', augmentation=aug_train_bool)
#     valset = AFDataset('val', augmentation=False)
#     testset = AFDataset('test', augmentation=False)

#     trainloader = DataLoader(trainset, batch_size=BATCHSIZE_TR, shuffle=True, drop_last=True, num_workers=0)
#     validloader = DataLoader(valset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)
#     testloader = DataLoader(testset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         # print("Device: ", torch.cuda.get_device_name(device))
#         torch.cuda.empty_cache()
    
#     segnet = SegNet(bayes=bayes_bool, dropout=dropout).to(device)
    

#     # print(pms.summary(net, torch.zeros((BATCHSIZE, 3, 256, 256)).to(device), show_input=True, show_hierarchical=False))
#     if loss_type == 'wce':
#         weights = torch.tensor([1.0, 18.0]).to(device)
#         criterion = nn.CrossEntropyLoss(weight=weights)
#     elif loss_type == 'bce':
#         criterion = nn.CrossEntropyLoss()
#     elif loss_type == 'dice':
#         criterion = DiceLoss()
#     elif loss_type == 'dicebce':
#         criterion = DiceBCELoss()

#     optimizer = optim.SGD(segnet.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#     # optimizer = optim.Adam(segnet.parameters(), lr=LR, weight_decay=5e-4)

#     num_batch_train = len(trainloader) # 9
#     num_batch_val = len(validloader)
#     num_batch_test = len(testloader)

#     # test  
#     # print('num batch train: ', num_batch_train)
#     # # xxx = next(iter(trainloader))
#     # # while True:
#     # #     imgs, masks = next(iter(trainloader))
#     # xxx = None
#     # for i, data in cycle(enumerate(trainloader)):
#     #     print(i)
#     #     if (i+1)%9 == 0:
#     #         if xxx is not None and not torch.equal(xxx, data[0]):
#     #             print('Diff', i)
#     #         else:
#     #             print('Same', i)
#     #         xxx = data[0]
#     train_iter = iter(trainloader)
#     begin_data = None
#     for i in range(10):
#         try:
#             data = next(train_iter)
#         except StopIteration:
#             train_iter = iter(trainloader)
#             data = next(train_iter)
#             if begin_data is not None:
#                 if torch.equal(begin_data, data[0]):
#                     print('Same')
#                 else:
#                     print('Diff')
            
#             begin_data = data[0]



# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import numpy as np
# import math
# # a = [0,1,2, 3]
# # b = [1, 2, 3, 4]
# # # for ind, (i, j) in enumerate(zip(a, b)):
# # #     print(ind, i, j) 

# # for i in range(0, len(b), 2):
# #     print(b[i])


# # fig, axs = plt.subplots(1, 2)
# # axs[0].plot(a, b)
# # axs[1].plot(a, b)
# # fig.set_title('ddd')
# # fig.tight_layout()
# # plt.show()
# # m = nn.Dropout(p=0.8)
# # input = torch.randn(5, 5)
# # output = m(input)
# # print(input)
# # print(output)
# # if not False:
# #     print('True')
# # print(np.version.version)

# # a = np.arange(48).reshape(1, 2, 4, 6)
# # b = a.argmax(axis=1)
# # xxx = np.take_along_axis(a, b[:,None],axis=1)[:,0]
# # print(a.shape)
# # print(b.shape)
# # print(xxx.shape)
# # print(a)
# # print(b)
# # print(xxx)


# # black_img = np.zeros(shape=(1,2,256,256))
# # black_img[0,0,:,:] += 1
# # pred = black_img.argmax(axis=1)
# # target = np.zeros(shape=(1,256,256))
# # target[0,100:150,100:150] += 1

# # weights = torch.tensor([18.0, 1.0])
# # criterion_weighted = nn.CrossEntropyLoss(weight=weights)
# # criterion = nn.CrossEntropyLoss()

# # black_img = torch.tensor(black_img, dtype=torch.float32)
# # target = torch.tensor(target, dtype=torch.long)

# # loss = criterion(black_img, target) 
# # loss_weighted = criterion_weighted(black_img, target)

# # print(loss_weighted > loss)
# x = torch.arange(24).reshape((2,3,4))
# print(x)
# x_reshaped = torch.flatten(x)
# print(x_reshaped)
# x_restore = x_reshaped.reshape((2,3,4))
# print(x_restore)

# print()









# p_same = 0.9
# p_diff = 0.1
# channels = 10
# weight_cs = np.array([[p_same, p_diff],[p_diff, p_same]])
# weights = np.repeat(weight_cs[np.newaxis, :, :], channels, axis=0)
        
# print()




# '''
# Self defined linear layer
# '''
# class MyLinearLayer(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """
#     def __init__(self, size_in, size_out):
#         super().__init__()
#         self.size_in, self.size_out = size_in, size_out
#         weights = torch.Tensor(size_out, size_in)
#         self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
#         bias = torch.Tensor(size_out)
#         self.bias = nn.Parameter(bias)

#         # initialize weights and biases
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in)
#         nn.init.uniform_(self.bias, -bound, bound)  # bias init

#     def forward(self, x):
#         xxx = self.weights.t()
#         w_times_x= torch.mm(x, self.weights.t())  # t() transform
#         return torch.add(w_times_x, self.bias)  # w times x + b


# class BasicModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(1, 128, 3)
#         # self.linear = nn.Linear(256, 2)
#         self.linear = MyLinearLayer(256, 2)

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(-1, 256)
#         return self.linear(x)



# torch.manual_seed(0)  #  for repeatable results
# basic_model = BasicModel()
# inp = np.array([[[[1,2,3,4],  # size (1, 1, 3, 4)
#                   [1,2,3,4],
#                   [1,2,3,4]]]])
# x = torch.tensor(inp, dtype=torch.float)
# print('Forward computation thru model:', basic_model(x))
























