from cProfile import label
import os
import subprocess
import random
import shutil
from PIL import Image
from skimage.measure import label, regionprops
import scipy

from os.path import join, split
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap   
from matplotlib.pyplot import cm, Normalize 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import time
import re
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from torchsummary import summary

### AF SLO CLASSIFICATION ###
CLA_DATA_OLD_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\SLO_AF_classifier\Oxford'
# allocate to the new directory
CLA_DATA_NEW_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\SLO_AF_classifier\data_new'
CLA_MODEL_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\SLO_AF_class\code\classifier'
CLA_H5_FILE_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\SLO_AF_classifier\hdf5'


### AF SEGMENTATION ###
SEG_DATA_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\data_raw"
RPGR_PATH = join(SEG_DATA_PATH, 'RPGR')
USH_PATH = join(SEG_DATA_PATH, 'USH')

SEG_H5_FILE_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\hdf5"
SEG_MODEL_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\AF_Seg\code\model\local"
SEG_VISUAL_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\AF_Seg\code\visual"


### AF SLO MTL ###
MTL_MODEL_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\MTL\model"
MTL_VISUAL_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\MTL\visual"

NORM_SIZE_CLA = 256

def data_extract(indices, x_all, y_all):
    xs = []
    ys = []
    for index in indices:
        xs.append(x_all[index])
        ys.append(y_all[index])
    
    return xs, ys

# def data_split(indice, image_all, mask_all):
#     imgs = []
#     masks = []
#     for id in indice:
#         imgs.append(image_all[id])
#         masks.append(mask_all[id])
#     return imgs, masks


def stat_compute_log(stat_list, str_type, folds=5):
    stat_avg = np.mean(stat_list)
    stat_std = np.std(stat_list)

    stat_list = [round(num,4) for num in stat_list]
    print(f'{folds}-fold {str_type}: ', stat_list)
    print('%d-fold %s mean and std: %.4f \u00B1 %.4f' % (folds, str_type, stat_avg, stat_std))

    return stat_avg, stat_std




'''
AF Segmentation helper functions
'''

# inputs(1,3,768,768) outputs(1,3,768,768)
def patch_infer_output(model, inputs, norm_size, device):
    stride = 128
    psize = 256
    gsize = (norm_size-psize)//stride+1
    
    outputs = torch.zeros((1,2,norm_size,norm_size)).to(device)
    count = torch.zeros((norm_size, norm_size)).to(device)
    for i in range(gsize):
        for j in range(gsize):
            x_begin = i*stride
            x_end = x_begin + psize
            y_begin = j*stride
            y_end = y_begin + psize

            patch = TF.crop(inputs, x_begin, y_begin, psize, psize)
            poutput = model(patch)
            outputs[...,x_begin:x_end,y_begin:y_end] += poutput 
            count[x_begin:x_end,y_begin:y_end] += 1

    outputs = torch.div(outputs, count)
    return outputs


def clahe_apply(image):
    clipLimit = 2.0
    gridsize = 8
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridsize,gridsize))
    g,b,r = cv2.split(image)
    g = clahe.apply(g)
    b = clahe.apply(b)
    r = clahe.apply(r)
    img_clahe = cv2.merge((g,b,r))

    return img_clahe



def post_process(pred_mask, norm_size):
    labeled_img = label(pred_mask, connectivity=2) 
    regions = regionprops(labeled_img) # not include background
    num_region = len(regions)
    if num_region == 0: 
        pass
    else:
        # the distance between region centroid and image center
        dst_list = []
        area_list = []
        for region in regions:
            re_cen = np.array(region.centroid)
            img_cen = np.array([norm_size/2,norm_size/2])
            dst = np.linalg.norm(re_cen-img_cen)
            dst_list.append(dst)
            area_list.append(region.area)
        
        if num_region < 3:
            idx_best = np.argmin(dst_list)
        else:
            # indices of 3 max areas
            idx_maxarea_candid = np.argpartition(area_list, -3)[-3:]
            idx_maxarea = []
            for i in idx_maxarea_candid:
                if area_list[i] > 100:
                    idx_maxarea.append(i)
            
            # region index with their corresponding distance
            if len(idx_maxarea) == 0:
                dst_candid = [dst_list[i] for i in idx_maxarea_candid]
            else:    
                dst_candid = [dst_list[i] for i in idx_maxarea]
            idx_best = idx_maxarea[np.argmin(dst_candid)]

        pred_mask[labeled_img != idx_best+1] = 0
        pred_mask = scipy.ndimage.binary_fill_holes(pred_mask)
    
    return pred_mask

def load_imgs_seg(data_paths, data_type, norm_size, clahe_bool):
    """Crop and downsample images

    Args:
        data_paths: data paths 
        data_type: data type
        clahe_bool: CLAHE enhancement or not

    Returns:
        uniform size data
    """
    assert data_type == 'image' or 'mask', 'No such data type'
    images = []
    for img_path in data_paths:
        img = cv2.imread(img_path)

        if data_type == 'mask':
            # make mask values {0,255}
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = img//255

        # crop and resize image to the same size
        x, y, _ = img.shape
        length = min(x, y)
        
        if x != y:
            img = img[:length, :length]
        
        if clahe_bool and data_type == 'image':
            img = clahe_apply(img)

        if img.shape[0] != norm_size:
            img = cv2.resize(img, (norm_size, norm_size))
    
        # img = img.astype(np.uint8)
        images.append(np.array(img))

    return np.array(images)



def load_imgs_cla(img_paths):
    """Crop/downsample and load image data into an array of images 

    Args:
        img_paths: image paths

    Returns:
        an array of images
    """
    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        # crop and resize image to the same size
        x, y, _ = img.shape
        if x != y:
            length = min(x, y)
            img = img[:length, :length]
        if img.shape[0] != NORM_SIZE_CLA:
            img = cv2.resize(img, (NORM_SIZE_CLA, NORM_SIZE_CLA))
        
        images.append(np.array(img))

    return np.array(images)


def image_restore_seg(img, size_origin):
    """Restore a single mask: upsample and add all black if original mask is not square

    Args:
        imgs: downsampled mask predictions
        size_origin: original mask size
    """
    if torch.is_tensor(img):
        img = img.cpu().detach().numpy()
    x, y = size_origin
    length = min(x, y)
    img = cv2.resize(np.array(img, dtype=np.uint8), (length, length))
    if x != y:
        assert x>y
        img = cv2.copyMakeBorder(img, 0, x-y, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)

    return img



def create_h5_train_seg(imgs, masks, type, norm_size, clahe):
    """Create the hdf5 file for each subset containing images and masks

    Args:
        imgs: a list of train image paths
        masks: a list of train mask paths
    """
    imgs_arr = load_imgs_seg(imgs, 'image', norm_size, clahe)
    masks_arr = load_imgs_seg(masks, 'mask', norm_size, clahe)

    h5_file_name = join(SEG_H5_FILE_PATH, f'{type}.h5')
    if os.path.exists(h5_file_name): os.remove(h5_file_name)

    with h5py.File(h5_file_name, 'w') as f:
        f.create_dataset(name='imgs', data=imgs_arr)
        f.create_dataset(name='masks', data=masks_arr)


def create_h5_test_seg(imgs, masks, type, norm_size, clahe, fid=None):
    """Create the hdf5 file for each subset

    Args:
        imgs: a list of val/test image paths
        masks: a list of val/test mask paths
        type: 'val' or 'test'
    """
    # save the resized images and masks
    imgs_arr = load_imgs_seg(imgs, 'image', norm_size, clahe)
    masks_arr = load_imgs_seg(masks, 'mask', norm_size, clahe)

    if type == 'val':
        h5_file_name = join(SEG_H5_FILE_PATH, 'val.h5')
        if os.path.exists(h5_file_name): os.remove(h5_file_name)
    elif type == 'test':
        h5_file_name = join(SEG_H5_FILE_PATH, f'test{fid}.h5')

    with h5py.File(h5_file_name, 'w') as f:
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            img_name = os.path.split(img)[-1].split('.')[0]
            mask_name = os.path.split(mask)[-1].split('.')[0]
            assert img_name == mask_name, "image and mask don't match"

            # test
            # if img_name == '1564100_OD_2016':
            #     print(img_name)
            #     xxx = cv2.imread(mask)
            #     _, xxx = cv2.threshold(xxx, 127, 255, cv2.THRESH_BINARY)
            #     xxx = xxx//255*255
                
            #     cv2.imshow('xxx', xxx)
            #     print()

            mask_origin = cv2.imread(mask)
            _, mask_origin = cv2.threshold(mask_origin, 127, 255, cv2.THRESH_BINARY)
            mask_origin = mask_origin//255
            img_trans = imgs_arr[i]
            mask_trans = masks_arr[i]
        
            f.create_dataset(name=img_name, data=np.array([img_trans, mask_trans]))
            f.create_dataset(name=img_name+'_origin', data=np.array(mask_origin, dtype='uint8'))




 
def seg_visual(pred_mask_restore, var_map, miou_restore, img_name, fig_name, model_type):
    """Display image, ground truth, restored predicted mask, uncertainty map 

    Args:
        pred_mask_restore: restored predicted mask
        var_map: uncertainty map
        miou_restore: mIoU score of the restored mask
        img_name: image filename
        fig_name: saved filename
    """
    assert model_type == 'mtl' or 'seg'
    if type(pred_mask_restore) != np.ndarray:
        pred_mask_restore = pred_mask_restore.cpu().detach().numpy()
    
    if var_map is None:
        fig, axs = plt.subplots(1, 3)
    else:
        fig, axs = plt.subplots(1, 4)

    if 'RPGR' in img_name:
        img_path = join(RPGR_PATH, img_name+'.tif')
        gt_path = join(RPGR_PATH, img_name+'.jpg')
    else:
        img_path = join(USH_PATH, img_name+'.tif')
        gt_path = join(USH_PATH, img_name+'.jpg')
    
    img = plt.imread(img_path)
    gt = plt.imread(gt_path)
    pred_mask_restore = np.repeat(pred_mask_restore[:, :, np.newaxis], 3, axis=2)*255
    
    axs[0].imshow(img)    
    axs[0].axis('off')
    
    axs[1].imshow(gt)
    axs[1].set_title('Ground truth mask')
    axs[1].axis('off')
    
    axs[2].imshow(pred_mask_restore)
    axs[2].set_title('Predicted mask \n (mIoU=%.3f)' % miou_restore)
    axs[2].axis('off')

    if var_map is not None:
        axs[3].imshow(var_map, cmap='Greys')
        axs[3].set_title('Uncertainty \n map')
        axs[3].axis('off')

    plt.tight_layout()
    if model_type == 'mtl':
        plt.savefig(join(MTL_VISUAL_PATH, fig_name))  
    else:
        plt.savefig(join(SEG_VISUAL_PATH, fig_name))
    plt.close()

    # # test
    # save_dir_path = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\AF_Seg\visual_demo'
  
    # shutil.copy(img_path, save_dir_path) 
    # if var_map is not None:
    #     cv2.normalize(var_map, var_map, 0, 255, cv2.NORM_MINMAX)
    #     var_map = np.repeat(var_map[:, :, np.newaxis], 3, axis=2)
    #     cv2.imwrite(join(save_dir_path, img_name+'_probmap.png'), var_map)



# All images are of size (256, 256, 3)
def seg_visual_overlap(norm_size, gt_mask, pred_mask, var_map, img_name, fig_name, model_type):
    """Display downsampled image, ground truth (boundary), 
               restored predicted mask (boundary), uncertainty map (heatmap) in a same image 

    Args:
        pred_mask_restore: restored predicted mask
        var_map: uncertainty map
        miou_restore: mIoU score of the restored mask
        img_name: image filename
        fig_name: saved filename
    """
    assert model_type == 'mtl' or 'seg'
    
    # read original image
    if 'RPGR' in img_name:
        img_path = join(RPGR_PATH, img_name+'.tif')
    else:
        img_path = join(USH_PATH, img_name+'.tif')
    image = cv2.imread(img_path)
    
    x, y, _ = image.shape
    if x != y:
        length = min(x, y)
        image = image[:length, :length]

    if image.shape[0] != norm_size:
        image = cv2.resize(image, (norm_size, norm_size))

    if type(gt_mask) != np.ndarray:
        gt_mask = gt_mask.cpu().detach().numpy() # (512,512)
    if type(pred_mask) != np.ndarray:
        pred_mask = pred_mask.cpu().detach().numpy() # (512,512)
    gt_mask = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2).astype('uint8')*255
    pred_mask = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2).astype('uint8')*255
    
    cmap = cm.YlOrBr
    c_cmap = cmap(np.arange(cmap.N))
    c_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    c_cmap = ListedColormap(c_cmap)

    if var_map is not None:
        norm = Normalize(vmin=var_map.min(), vmax=var_map.max())
        var_heatmap = c_cmap(norm(var_map))

        # Blend image with heatmap
        var_heatmap = cv2.cvtColor(np.uint8(var_heatmap * 255), cv2.COLOR_RGBA2BGRA)
        alpha = var_heatmap[..., 3] / 255
        alpha = np.tile(np.expand_dims(alpha, axis=2), [1, 1, 3])
        image = (image * (1 - alpha) + var_heatmap[..., :3] * alpha).astype(np.uint8)


    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)

    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, gt_contours, -1, (0,255,0), thickness=1, lineType=cv2.LINE_AA)
    # # test
    # if img_name == '1564100_OD_2016':
    #     cv2.imshow('gtmask', image)

    cv2.drawContours(image, pred_contours, -1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
    if model_type == 'mtl':
        cv2.imwrite(join(MTL_VISUAL_PATH, 'overlap_' + fig_name), image)
    else:
        cv2.imwrite(join(SEG_VISUAL_PATH, 'overlap_' + fig_name), image)

    # # test
    # save_dir_path = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\AF_Seg\visual_demo'
    # cv2.imwrite(join(save_dir_path, img_name+'_pred.png'), image)


def mIoU_score(pred, target, num_classes=2):
    """Compute the mIoU scores of mask pair

    Args:
        pred: _description_
        target: _description_
        num_classes: _description_. Defaults to 2.

    Returns:
        mIoU value
    """
    if type(pred) != np.ndarray:
        pred = pred.cpu().detach().numpy()
    if type(target) != np.ndarray:
        target = target.cpu().detach().numpy()
    
    inter1 = np.sum(np.multiply(pred, target))
    inter0 = np.sum(np.multiply(1-pred, 1-target))
    iou1 = inter1 / (np.sum(pred) + np.sum(target) - inter1 + 1e-8)
    iou0 = inter0 / (np.sum(1-pred) + np.sum(1-target) - inter0 + 1e-8)
    miou = (iou1+iou0) / 2

    return miou


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)[:,1,:,:]
        
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=1)[:,1,:,:]
        targets = targets.type(torch.float32)       
        
        #flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

# Multi-task learning loss: 
# Sum of classification loss (ce) and segmentation loss (dice-bce)
class MTLLoss(nn.Module):
    def __init__(self):
        super(MTLLoss, self).__init__()

    def forward(self, seg_inputs, cla_inputs, seg_targets, cla_targets, smooth=1):
        # seg_inputs (N,2,256,256)  seg_targets (N, 256, 256) 
        # cla_inputs (N,2)          cla_targets (N)
        seg_inputs = F.softmax(seg_inputs, dim=1)[:,1,:,:]
        seg_targets = seg_targets.type(torch.float32)       
        
        cla_inputs = F.softmax(cla_inputs, dim=1)[:,1]
        cla_targets = cla_targets.type(torch.float32)       
        
        # Flatten label and prediction tensors
        seg_inputs = seg_inputs.contiguous().view(-1)
        seg_targets = seg_targets.contiguous().view(-1)
        
        # cla_inputs = cla_inputs.contiguous().view(-1)
        # cla_targets = cla_targets.contiguous().view(-1)
        
        intersection = (seg_inputs * seg_targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(seg_inputs.sum() + seg_targets.sum() + smooth)  
        seg_bce = F.binary_cross_entropy(seg_inputs, seg_targets, reduction='mean')
        cla_bce = F.binary_cross_entropy(cla_inputs, cla_targets, reduction='mean')
        
        return dice_loss+seg_bce+cla_bce 




'''
AF/SLO Classification helper functions
'''


# def load_img(img_path):

#     img = cv2.imread(img_path)
#     # crop and resize image to the same size
#     NORM_SIZE = 768
#     x, y, _ = img.shape
#     if x != y:
#         length = min(x, y)
#         img = img[:length, :length]
#     if img.shape[0] != NORM_SIZE:
#         img = cv2.resize(img, (NORM_SIZE, NORM_SIZE))
    
#     return np.array(img)

def display_imgs_cla(imgs):
    num = 10
    fig, axs = plt.subplots(1, num)
    for i in range(num):
        axs[i].imshow(imgs[i])
        axs[i].axis('off')
    plt.savefig('img_display.png')


def create_h5_cla(imgs, labels, type, fid=None):
    """Create the hdf5 file for each subset containing images and labels

    Args:
        imgs: a list of image paths
        labels: a list of image labels
        type: 'train' or 'val' or 'test'
    """
    assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'
    imgs_arr = load_imgs_cla(imgs)
    labels_arr = np.array(labels)
    if type == 'test':
        h5_file_name = join(CLA_H5_FILE_PATH, f'{type}{fid}.h5')
    else:
        h5_file_name = join(CLA_H5_FILE_PATH, f'{type}.h5')
        if os.path.exists(h5_file_name): os.remove(h5_file_name)

    with h5py.File(h5_file_name, 'w') as f:
        f.create_dataset(name='imgs', data=imgs_arr)
        f.create_dataset(name='labels', data=labels_arr)


def change_state_keys(state_dict, type):
    assert type == 'seg' or 'cla'
    old_keys = list(state_dict.keys())
    for old_key in old_keys:
        if type == 'seg':
            rexpr = r'conv_block\1_1'
        else:
            rexpr = r'conv_block\1_2'
        new_key = re.sub(r'^conv_block(\d)', rexpr, old_key)
        state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


def roc_plot(fpr, tpr, auc, fid):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for SLO AF classification")
    plt.legend(loc="lower right")
    plt.savefig(join(CLA_MODEL_PATH, f'roc_curve_fold{fid+1}.png'))


def roc_plot_all(fprs, tprs, auc_mean, auc_std, type):
    # fpr_mean = np.mean(fprs, axis=0)
    # tpr_mean = np.mean(tprs, axis=0)

    plt.figure()
    
    # all curves
    fair_color_list = ["coral", "gold", "yellowgreen", "turquoise", "darkorchid"]
    for i in range(5):
        plt.plot(fprs[i], tprs[i], color=fair_color_list[i], lw=1, label=f"fold {i+1}")
    plt.plot([0, 1], [0, 1], color="lightgray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves for all folds (AUC = %0.4f \u00B1 %0.4f)" % (auc_mean, auc_std))
    plt.legend(loc="lower right")
    plt.savefig(join(SEG_MODEL_PATH, f'{type}_ROC_curves_all.png'))
