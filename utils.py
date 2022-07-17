import os
import random

from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from torch.utils.data import Dataset
import h5py

### AF SLO CLASSIFICATION ###
CLA_DATA_OLD_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\SLO_AF_classifier\Oxford'
# allocate to the new directory
CLA_DATA_NEW_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\SLO_AF_classifier\data_new'
CLA_MODEL_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\SLO_AF_class\code\classifier'
CLA_H5_FILE_PATH = r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\SLO_AF_classifier\hdf5'


### AF CLASSIFICATION ###
SEG_DATA_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\data_raw"
RPGR_PATH = join(SEG_DATA_PATH, 'RPGR')
USH_PATH = join(SEG_DATA_PATH, 'USH')

SEG_H5_FILE_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\hdf5"
SEG_MODEL_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\AF_Seg\code\model"
SEG_VISUAL_PATH = r"C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\AF_Seg\code\visual"


### AF SLO MTL ###








'''
AF Segmentation helper functions
'''
def load_imgs_seg(data_paths, data_type, transform=True):
    """transform images or masks into an array of images with uniform image size

    Args:
        data_paths: data paths 
        data_type: data type
        transform: transform data or not. Defaults to True.

    Returns:
        uniform size data
    """
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



def image_restore_seg(imgs, size_origin):
    """Restore the image to original image size (upsample, crop)

    Args:
        img: numpy array
    """
    if torch.is_tensor(imgs):
        imgs = imgs.cpu().detach().numpy()
    batch_len, x, y = size_origin
    imgs_origin = []
    for i in range(batch_len):
        length = max(x, y)
        img = cv2.resize(np.array(imgs[i], dtype=np.uint8), (length, length))
        if x < y:
            img = img[:x,:]
        if x > y:
            img = img[:,:y]
        imgs_origin.append(img)
    
    return np.array(imgs_origin)



def create_h5_train_seg(imgs, masks, type):
    """Create the hdf5 file for each subset containing images and masks

    Args:
        imgs: a list of train image paths
        masks: a list of train mask paths
    """
    trans = True
    imgs_arr = load_imgs_seg(imgs, 'image', transform=trans)
    masks_arr = load_imgs_seg(masks, 'mask', transform=trans)

    h5_file_name = join(SEG_H5_FILE_PATH, f'{type}.h5')
    with h5py.File(h5_file_name, 'w') as f:
        f.create_dataset(name='imgs', data=imgs_arr)
        f.create_dataset(name='masks', data=masks_arr)


def create_h5_test_seg(imgs, masks, type):
    """Create the hdf5 file for each subset

    Args:
        imgs: a list of val/test image paths
        masks: a list of val/test mask paths
        type: 'val' or 'test'
    """
    # save the resized images and masks
    imgs_arr = load_imgs_seg(imgs, 'image')
    masks_arr = load_imgs_seg(masks, 'mask')

    # name_list = []
    h5_file_name = join(SEG_H5_FILE_PATH, f'{type}.h5')
    with h5py.File(h5_file_name, 'w') as f:
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            img_name = os.path.split(img)[-1].split('.')[0]
            mask_name = os.path.split(mask)[-1].split('.')[0]
            assert img_name == mask_name, "image and mask don't match"
            
            mask_origin = cv2.imread(mask)/255
            img_trans = imgs_arr[i]
            mask_trans = masks_arr[i]
        
            f.create_dataset(name=img_name, data=np.array([img_trans, mask_trans]))
            f.create_dataset(name=img_name+'_origin', data=np.array(mask_origin, dtype='uint8'))
            # name_list.append(mask_name)
    # print(f'name list for {type} before creating hdf5: {name_list}')


# def display_image(imgs, filename):
#     if type(imgs) != np.ndarray:
#         imgs = imgs.cpu().detach().numpy()
#     img = np.repeat(imgs[0][:, :, np.newaxis], 3, axis=2).astype('uint8')*255
#     plt.imsave(join(VISUAL_PATH, filename), img)


# def display_images(imgs, filename):
#     if type(imgs) != np.ndarray:
#         imgs = imgs.cpu().detach().numpy()
#     num = imgs.shape[0]
#     fig, axs = plt.subplots(1, num)
#     for i in range(num):
#         img = np.repeat(imgs[i][:, :, np.newaxis], 3, axis=2)*255
#         axs[i].imshow(img)
#         axs[i].axis('off')
#     plt.savefig(join(VISUAL_PATH, filename))



def seg_visual(pred_mask_ds, pred_mask_res, var_map, miou_ds, miou_res, img_name, fig_name, uncertainty_analysis=False):
    """Display image, ground truth, downsampled predicted mask, restored predicted mask, uncertainty map 

    Args:
        pred_mask: predicted mask
        img_name: image filename
    """
    if type(pred_mask_ds) != np.ndarray:
        pred_mask_ds = pred_mask_ds.cpu().detach().numpy()
    if type(pred_mask_res) != np.ndarray:
        pred_mask_res = pred_mask_res.cpu().detach().numpy()
    
    fig, axs = plt.subplots(1, 5)
    
    if 'RPGR' in img_name:
        img_path = join(RPGR_PATH, img_name+'.tif')
        gt_path = join(RPGR_PATH, img_name+'.jpg')
    else:
        img_path = join(USH_PATH, img_name+'.tif')
        gt_path = join(USH_PATH, img_name+'.jpg')
    
    img = plt.imread(img_path)
    gt = plt.imread(gt_path)
    pred_mask_ds = np.repeat(pred_mask_ds[:, :, np.newaxis], 3, axis=2)*255
    pred_mask_res = np.repeat(pred_mask_res[:, :, np.newaxis], 3, axis=2)*255
    
    axs[0].imshow(img)    
    axs[0].axis('off')
    
    axs[1].imshow(gt)
    axs[1].set_title('Ground truth')
    axs[1].axis('off')
    
    axs[2].imshow(pred_mask_ds)
    axs[2].set_title('Downsampled \n (mIoU=%.3f)' % miou_ds)
    axs[2].axis('off')
    
    axs[3].imshow(pred_mask_res)
    axs[3].set_title('Restored \n (mIoU=%.3f)' % miou_res)
    axs[3].axis('off')

    if uncertainty_analysis:
        assert var_map is not None
        axs[4].imshow(var_map, cmap='Greys')
        axs[4].set_title('Uncertainty \n map')
        axs[4].axis('off')

    plt.tight_layout()
    plt.savefig(join(SEG_VISUAL_PATH, fig_name))  
    plt.close()


def mIoU_score(preds, targets, num_classes=2):
    """Compute the sum of the mIoU scores of a batch of data

    Args:
        preds: _description_
        targets: _description_
        num_classes: _description_. Defaults to 2.

    Returns:
        _description_
    """
    if type(preds) != np.ndarray:
        preds = preds.cpu().detach().numpy()
    if type(targets) != np.ndarray:
        targets = targets.cpu().detach().numpy()
    batch_len = len(targets)
    mIoUs = np.empty(batch_len)
    for i in range(batch_len):
        pred = preds[i]
        target = targets[i]
        inter1 = np.sum(np.multiply(pred, target))
        inter0 = np.sum(np.multiply(1-pred, 1-target))
        iou1 = inter1 / (np.sum(pred) + np.sum(target) - inter1 + 1e-8)
        iou0 = inter0 / (np.sum(1-pred) + np.sum(1-target) - inter0 + 1e-8)
        miou = (iou1+iou0) / 2
        mIoUs[i] = miou

    return np.sum(mIoUs)


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
# transform image data into an array of images 
# image shape (1024, 1024, 3)
def load_imgs_cla(img_paths):
    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        # crop and resize image to the same size
        norm_size = 256
        x, y, _ = img.shape
        if x != y:
            length = min(x, y)
            img = img[:length, :length]
        if img.shape[0] != norm_size:
            img = cv2.resize(img, (norm_size, norm_size))
        
        images.append(np.array(img))

    return np.array(images)

# def load_img(img_path):

#     img = cv2.imread(img_path)
#     # crop and resize image to the same size
#     norm_size = 768
#     x, y, _ = img.shape
#     if x != y:
#         length = min(x, y)
#         img = img[:length, :length]
#     if img.shape[0] != norm_size:
#         img = cv2.resize(img, (norm_size, norm_size))
    
#     return np.array(img)

def display_imgs_cla(imgs):
    num = 10
    fig, axs = plt.subplots(1, num)
    for i in range(num):
        axs[i].imshow(imgs[i])
        axs[i].axis('off')
    plt.savefig('img_display.png')


def create_h5_cla(imgs, labels, type):
    """Create the hdf5 file for each subset containing images and labels

    Args:
        imgs: a list of image paths
        labels: a list of image labels
        type: 'train' or 'val' or 'test'
    """
    assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'
    imgs_arr = load_imgs_cla(imgs)
    labels_arr = np.array(labels)

    h5_file_name = join(CLA_H5_FILE_PATH, f'{type}.h5')
    with h5py.File(h5_file_name, 'w') as f:
        f.create_dataset(name='imgs', data=imgs_arr)
        f.create_dataset(name='labels', data=labels_arr)

