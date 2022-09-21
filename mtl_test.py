import os

from utils import *
from mtl_model import MTL
from dataset import AFDataset, ASDataset
from os.path import join, split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import h5py
import time

BATCHSIZE_TE_SEG = 1
BATCHSIZE_TE_CLA = 20
p_same = 0.9

cla_transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(256),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


segset = AFDataset('val', augmentation=False)
segloader = DataLoader(segset, batch_size=BATCHSIZE_TE_SEG, shuffle=False, num_workers=0)

claset = ASDataset('val', cla_transform_test)
claloader = DataLoader(claset, batch_size=BATCHSIZE_TE_CLA, shuffle=True, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # print("Device: ", torch.cuda.get_device_name(device))
    torch.cuda.empty_cache()

model_path = join(MTL_MODEL_PATH, 'fold1_epoch58_0.806_lite.pt')
mtl = MTL(device=device, p_same=p_same).to(device)
criterion = MTLLoss()
# mtl.load_state_dict(torch.load(), strict=False)
mtl.load_state_dict(torch.load(model_path))
mtl.eval()

print('Testing...')
with torch.no_grad():
    miou_mean_all = []
    miou_std_all = []
    loss_mean_all = []
    loss_std_all = []

    cla_iter = iter(claloader)
    for batch_idx, (seg_inputs, seg_targets, seg_targets_origins, seg_filenames) in enumerate(segloader):
        
        cla_inputs, cla_targets = next(cla_iter)
        seg_inputs, cla_inputs = seg_inputs.to(device), cla_inputs.to(device)
        seg_targets, cla_targets = seg_targets.to(device), cla_targets.to(device) 
        # 1 seg to 20 clas making up of 20 (seg, cla) pairs

        loss_pairs = []
        miou_pairs = []
        for run in range(BATCHSIZE_TE_CLA):
            cla_inputs_one = cla_inputs[run][None,:,:,:]
            cla_targets_one = cla_targets[run][None]
            seg_outputs, cla_outputs = mtl(seg_inputs, cla_inputs_one)
    
            loss = criterion(seg_outputs, cla_outputs, seg_targets, cla_targets_one)
            loss_pairs.append(loss.item())
            
            seg_preds = seg_outputs.argmax(dim=1)
            miou_pairs.append(mIoU_score(seg_preds, seg_targets)) 


        loss_pairs_mean = np.nanmean(loss_pairs)
        loss_pairs_std = np.nanstd(loss_pairs)

        miou_pairs_mean = np.nanmean(miou_pairs)
        miou_pairs_std = np.nanstd(miou_pairs)

        miou_mean_all.append(round(miou_pairs_mean,4))
        miou_std_all.append(round(miou_pairs_std,4))
        loss_mean_all.append(round(loss_pairs_mean,4))
        loss_std_all.append(round(loss_pairs_std,4))

    print('miou mean: ', miou_mean_all)   
    print('miou std: ', miou_std_all)
    print('loss mean: ', loss_mean_all)   
    print('loss std: ', loss_std_all)   

