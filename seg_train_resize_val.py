import os
import random

from utils import *
from SegNet import SegNet
from dataset import AFDataset
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
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import KFold
import h5py
import time




if __name__ == '__main__':

    save_best_model = True
    test_ratio = 0.2
    N_FOLD = 5
    BATCHSIZE_TR = 4
    BATCHSIZE_TE = 1
    LR = 0.001
    EPOCH = 1    
    aug_train_bool = False
    
    bayes_bool = False
    dropout = 0.2
    uncertain_ana = False 
    num_runs_bayes = 20
    
    loss_type = 'dicebce'


    print('Loading data...')
    img_paths = []
    mask_paths = []
    for root, dirs, files in os.walk(SEG_DATA_PATH):
        if len(files) != 0:
            for file in files:
                if '.tif' in file: 
                    img_paths.append(os.path.join(root, file))
                elif '.jpg' or '.gif' in file:
                    mask_paths.append(os.path.join(root, file))
    
    img_paths = sorted(img_paths)
    mask_paths = sorted(mask_paths)
    img_paths, mask_paths = shuffle(img_paths, mask_paths)
    assert len(img_paths) == len(mask_paths), 'image and mask number don\'t match'
    N = len(img_paths)

    for i in range(len(img_paths)):
        img_name = os.path.split(img_paths[i])[-1].split('.')[0]
        mask_name = os.path.split(mask_paths[i])[-1].split('.')[0]
        assert img_name == mask_name, "image and mask don't match"

    ### Split dataset into training and test set
    split_idx = int(N*test_ratio)
    test_imgs = img_paths[:split_idx]
    test_masks = mask_paths[:split_idx]
    print(f'test size {len(test_imgs)}')

    create_h5_test_seg(test_imgs, test_masks, 'test')

    trv_imgs = img_paths[split_idx:]
    trv_masks = mask_paths[split_idx:]

    # miou of the last epoch
    val_miou_ds_per_fold = np.zeros(N_FOLD)
    val_miou_per_fold = np.zeros(N_FOLD)
    train_miou_per_fold = np.zeros(N_FOLD)
    # miou of the best epoch
    val_miou_best_per_fold = np.zeros(N_FOLD)

    kf = KFold(n_splits=N_FOLD)
    for k_i, (train_index, val_index) in enumerate(kf.split(trv_imgs)):

        print(f'\n{k_i+1} fold model')
        print(f'{k_i+1} fold train size: {len(train_index)} val size: {len(val_index)}')
        train_imgs = []
        train_masks = []
        for idx in train_index:
            train_imgs.append(trv_imgs[idx])
            train_masks.append(trv_masks[idx])
        
        val_imgs = []
        val_masks = []
        for idx in val_index:
            val_imgs.append(trv_imgs[idx])
            val_masks.append(trv_masks[idx])


        # create hdf5 file for each split
        print('Generating hdf5 file...')
        t1 = time.time()
        create_h5_train_seg(train_imgs, train_masks, 'train')
        create_h5_test_seg(val_imgs, val_masks, 'val')
        t2 = time.time()
        print('hdf5 file generation time: %.2f (min)' % ((t2-t1)/60))

        trainset = AFDataset('train', augmentation=aug_train_bool)
        valset = AFDataset('val', augmentation=False)
        testset = AFDataset('test', augmentation=False)

        trainloader = DataLoader(trainset, batch_size=BATCHSIZE_TR, shuffle=True, num_workers=0)
        validloader = DataLoader(valset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # print("Device: ", torch.cuda.get_device_name(device))
            torch.cuda.empty_cache()
        
        segnet = SegNet(bayes=bayes_bool, dropout=dropout).to(device)
        

        # print(pms.summary(net, torch.zeros((BATCHSIZE, 3, 256, 256)).to(device), show_input=True, show_hierarchical=False))
        if loss_type == 'wce':
            weights = torch.tensor([1.0, 18.0]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        elif loss_type == 'bce':
            criterion = nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            criterion = DiceLoss()
        elif loss_type == 'dicebce':
            criterion = DiceBCELoss()

        optimizer = optim.SGD(segnet.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.Adam(segnet.parameters(), lr=LR, weight_decay=5e-4)

        num_batch_train = len(trainloader)
        num_batch_val = len(validloader)
        num_batch_test = len(testloader)


        print('\nTraining...')
        # save the train/valid loss/metric downsampled valid metric for every epoch 
        history = torch.zeros((EPOCH, 5))
        val_miou_best = 0.65
        model_best_path = None
        for epoch in range(EPOCH):
            ################### Training ###################
            segnet.train()
            # accumulate loss and mIoU over batches
            train_loss = 0
            mIoU_sum = 0
            total = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Train Epoch {epoch+1}")
                for batch_idx, (inputs, targets, _, _) in enumerate(tepoch):
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()

                    outputs = segnet(inputs)
                    loss = criterion(outputs, targets)
                
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    mIoU_sum += mIoU_score(preds, targets)
                    total += targets.size(0)

                    tepoch.set_postfix(loss=train_loss/(batch_idx+1), mIoU=100.*mIoU_sum/total)
            
            history[epoch][0] = train_loss / num_batch_train
            history[epoch][1] = mIoU_sum / total
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ################### Validation ################### 
            segnet.eval()
            # test with dropout
            if bayes_bool and uncertain_ana:
                segnet.apply(apply_dropout)

            val_loss = 0
            mIoU_sum_val_ds = 0
            mIoU_sum_val = 0
            total_val = 0
            with torch.no_grad():
                with tqdm(validloader, unit="batch") as vepoch:
                    for batch_idx, (inputs, targets, targets_origins, filenames) in enumerate(vepoch):
                        vepoch.set_description(f"Valid Epoch {epoch+1}")
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        var_maps = [None]
                        
                        if bayes_bool and uncertain_ana:
                            # model uncertainty analysis
                            loss_bayes_total = []
                            probs_total = []
                            for run in range(num_runs_bayes):
                                outputs = segnet(inputs)
                     
                                loss = criterion(outputs, targets)
                                loss_bayes_total.append(loss.item())

                                probs = F.softmax(outputs, dim=1)
                                probs_total.append(probs.cpu().detach().numpy())

                            probs_mean = np.nanmean(probs_total, axis=0)
                            probs_var = np.var(probs_total, axis=0)

                            val_loss += np.nanmean(loss_bayes_total)
                            preds = probs_mean.argmax(axis=1)

                            # generate the uncertainty map (probs_var, preds)
                            var_maps = np.take_along_axis(probs_var, preds[:,None],axis=1)[:,0]
                        
                        else:
                            outputs = segnet(inputs)
        
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            preds = outputs.argmax(dim=1)


                        # restore to original mask image size then compute miou 
                        preds_restore = image_restore_seg(preds, targets_origins.shape)
                                                
                        mscore_ds = mIoU_score(preds, targets)
                        mscore = mIoU_score(preds_restore, targets_origins)
                        if epoch == EPOCH-1:
                            seg_visual(preds[0], preds_restore[0], var_maps[0], mscore_ds, mscore, filenames[0], f'fold{k_i+1}_val_pred{batch_idx+1}.png', uncertain_ana)

                        mIoU_sum_val_ds += mscore_ds
                        mIoU_sum_val += mscore
                        total_val += targets.size(0)

                        vepoch.set_postfix(loss=val_loss/(batch_idx+1), mIoU_downsample=100.*mIoU_sum_val_ds/total_val, mIoU=100.*mIoU_sum_val/total_val)
            
            history[epoch][2] = val_loss / num_batch_val
            history[epoch][3] = mIoU_sum_val / total_val
            history[epoch][4] = mIoU_sum_val_ds / total_val
            # Save the model(epoch) with highest val miou during training
            if save_best_model and history[epoch][3] > val_miou_best:
                if model_best_path is not None:
                    os.remove(model_best_path)
                modelpath = join(SEG_MODEL_PATH, 'fold%d_epoch%d_%.3f_lite.pt' % (k_i+1, epoch+1, history[epoch][3]))
                torch.save(segnet.state_dict(), modelpath)
                val_miou_best = history[epoch][3]
                model_best_path = modelpath

        train_miou_per_fold[k_i] = history[-1,1]
        val_miou_ds_per_fold[k_i] = history[-1,-1]
        val_miou_per_fold[k_i] = history[-1,-2]
        val_miou_best_per_fold[k_i] = val_miou_best
        if not save_best_model:
            torch.save(history, join(SEG_MODEL_PATH, f'history{k_i+1}.pt'))
            torch.save(segnet.state_dict(), join(SEG_MODEL_PATH, 'fold%d_%.3f_lite.pt' % (k_i+1, history[-1,-2])))


        ### plot training and validation history
        x = np.arange(EPOCH)
        plt.figure()
        plt.plot(x, history[:,0], label='train loss') # train loss
        plt.plot(x, history[:,2], label='val loss') # val loss
        plt.legend()
        plt.title('Training and validation loss for {} epochs'.format(EPOCH))
        plt.savefig(join(SEG_MODEL_PATH, f'train_val_loss{k_i+1}.png'))

        plt.figure()
        plt.plot(x, history[:,1], label='train mIoU') 
        plt.plot(x, history[:,3], label='val mIoU') 
        plt.plot(x, history[:,4], label='downsampled val mIoU') 
        plt.legend()
        plt.title('Training and validation mIoU for {} epochs'.format(EPOCH))
        plt.savefig(join(SEG_MODEL_PATH, f'train_val_miou{k_i+1}.png'))

        trainset.close()
        valset.close()
        testset.close()
        del train_imgs, train_masks, val_imgs, val_masks


    train_miou_avg = np.mean(train_miou_per_fold)
    train_miou_std = np.std(train_miou_per_fold)
    val_miou_ds_avg = np.mean(val_miou_ds_per_fold)
    val_miou_ds_std = np.std(val_miou_ds_per_fold)
    val_miou_avg = np.mean(val_miou_per_fold)
    val_miou_std = np.std(val_miou_per_fold)
    if save_best_model:
        val_miou_best_avg = np.mean(val_miou_best_per_fold)
        val_miou_best_std = np.std(val_miou_best_per_fold)

    print('\nTrain finished.')
    print(f'{N_FOLD}-fold train mIoU: ', train_miou_per_fold)
    print('%d-fold train mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, train_miou_avg, train_miou_std))
    
    print(f'\n{N_FOLD}-fold downsampled validation mIoU: ', val_miou_ds_per_fold)
    print('%d-fold downsampled validation mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, val_miou_ds_avg, val_miou_ds_std))
    
    print(f'\n{N_FOLD}-fold validation mIoU: ', val_miou_per_fold)
    print('%d-fold validation mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, val_miou_avg, val_miou_std))

    if save_best_model:
        print(f'\n{N_FOLD}-fold optimal validation mIoU: ', val_miou_best_per_fold)
        print('%d-fold optimal validation mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, val_miou_best_avg, val_miou_best_std))
    

