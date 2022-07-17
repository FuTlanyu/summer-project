'''
HDF5 fetch data implemented
'''

import os

from utils import *
from dataset import ASDataset
from sklearn.metrics import accuracy_score
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from Resnet import *
from tqdm import tqdm
from time import sleep
from sklearn.model_selection import KFold
import h5py
import time





if __name__ == '__main__':

    BATCHSIZE = 16
    LR = 0.1
    EPOCH = 1

    '''
    Fetch the data from disk only when necessary
    '''
    print('Loading data...')

    img_path_all = []
    label_all = []
    for root, dirs, files in os.walk(CLA_DATA_NEW_PATH):
        if len(files) != 0:
            for file in files:
                img_path_all.append(os.path.join(root, file))
                if 'AF' in file:
                    label_all.append(1)
                elif 'SLO' in file:
                    label_all.append(0)

 
    img_path_all, label_all = shuffle(img_path_all, label_all)
    assert len(img_path_all) == len(label_all), 'image and label size does not match'
    N = len(img_path_all)
    
    # split whole dataset into training and test set
    test_ratio = 0.1
    split_idx = int(N*test_ratio)
    test_imgs = img_path_all[:split_idx]
    test_labels = label_all[:split_idx]
    print(f'test size {len(test_imgs)}')
    
    create_h5_cla(test_imgs, test_labels, 'test')

    trv_imgs = img_path_all[split_idx:]
    trv_labels = label_all[split_idx:]

    
    transform_train = transforms.Compose(
        [transforms.ToPILImage(),
        # transforms.RandomCrop(512),
        transforms.RandomRotation((0,180)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    N_FOLD = 9
    val_acc_opt = 0
    valid_acc_ks = np.zeros(N_FOLD)

    kf = KFold(n_splits=N_FOLD)
    for k_i, (train_index, val_index) in enumerate(kf.split(trv_imgs)):

        print(f'\n{k_i+1} fold model')
        print(f'{k_i+1} fold train size: {len(train_index)} val size: {len(val_index)}')
        train_imgs = []
        train_labels = []
        for idx in train_index:
            train_imgs.append(trv_imgs[idx])
            train_labels.append(trv_labels[idx])
        
        val_imgs = []
        val_labels = []
        for idx in val_index:
            val_imgs.append(trv_imgs[idx])
            val_labels.append(trv_labels[idx])


        # create hdf5 file for each split
        print('Generating hdf5 file...')
        t1 = time.time()
        create_h5_cla(train_imgs, train_labels, 'train')
        create_h5_cla(val_imgs, val_labels, 'val')
        t2 = time.time()
        print('hdf5 file generation time: %.2f (min)' % ((t2-t1)/60))

        trainset = ASDataset('train', transform_train)
        valset = ASDataset('val', transform_test)
        testset = ASDataset('test', transform_test)

        trainloader = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
        validloader = DataLoader(valset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
        testloader = DataLoader(testset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # print("Device: ", torch.cuda.get_device_name(device))
            torch.cuda.empty_cache()
        
        resnet18 = ResNet18().to(device)
        # resnet34 = ResNet34().to(device)
        # resnet50 = ResNet50().to(device)

        # print(pms.summary(net, torch.zeros((BATCHSIZE, 3, 256, 256)).to(device), show_input=True, show_hierarchical=False))
        criterion = torch.nn.CrossEntropyLoss()
        
        # optimizer = optim.Adam(resnet18.parameters(), lr=LR)
        optimizer = optim.SGD(resnet18.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

        num_batch_train = len(trainloader)
        num_batch_val = len(validloader)
        num_batch_test = len(testloader)

        # imgs, labels = next(iter(trainloader))

        print('\nTraining...')
        # best_acc = 0
        # save the train/valid loss/metric for every epoch 
        history = torch.zeros((EPOCH, 4))
        for epoch in range(EPOCH):
            ################### Training ###################
            resnet18.train()
            # accumulate loss and accuarcy over batches
            train_loss = 0
            correct = 0
            total = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Train Epoch {epoch+1}")
                for batch_idx, (inputs, targets) in enumerate(tepoch):
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()

                    outputs = resnet18(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    _, preds = outputs.max(1)
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()
                
                    tepoch.set_postfix(loss=train_loss/(batch_idx+1), accuracy=100.*correct/total)
                    # sleep(0.1)
            # print(f'train epoch {epoch} total {total}')
            history[epoch][0] = train_loss / num_batch_train
            history[epoch][1] = correct / total
            scheduler.step(train_loss / num_batch_train)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ################### Validation ################### 
            resnet18.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                with tqdm(validloader, unit="batch") as vepoch:
                    for batch_idx, (inputs, targets) in enumerate(vepoch):
                        vepoch.set_description(f"Valid Epoch {epoch+1}")

                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = resnet18(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, preds = outputs.max(1)
                        total_val += targets.size(0)
                        correct_val += preds.eq(targets).sum().item()
                
                        vepoch.set_postfix(loss=val_loss/(batch_idx+1), accuracy=100.*correct_val/total_val)

            acc = 100.*correct_val/total_val
            # print(f'valid epoch {epoch} total {total_val}')
            
            history[epoch][2] = val_loss / num_batch_val
            history[epoch][3] = correct_val / total_val
            
            print('Training: loss: %.5f, accuracy: %.5f' % (history[epoch][0], history[epoch][1]))
            print('Validation: loss: %.5f, accuracy: %.5f' % (history[epoch][2], history[epoch][3]))
        
        valid_acc_ks[k_i] = history[-1,-1]
        if history[-1,-1] > val_acc_opt:
            torch.save(history, join(CLA_MODEL_PATH, f'history{k_i+1}.pt'))
            torch.save(resnet18.state_dict(), join(CLA_MODEL_PATH, f'class_model_lite{k_i+1}.pt'))
            val_acc_opt = history[-1,-1] 


        ### plot training and validation history
        x = np.arange(EPOCH)
        plt.figure()
        plt.plot(x, history[:,0], label='train loss') # train loss
        plt.plot(x, history[:,2], label='val loss') # val loss
        plt.legend()
        plt.title('Training and validation loss for {} epochs'.format(EPOCH))
        plt.savefig(join(CLA_MODEL_PATH, f'train_val_loss{k_i+1}.png'))

        plt.figure()
        plt.plot(x, history[:,1], label='train acc') # train acc
        plt.plot(x, history[:,3], label='val acc') # val acc
        plt.legend()
        plt.title('Training and validation accuracy for {} epochs'.format(EPOCH))
        plt.savefig(join(CLA_MODEL_PATH, f'train_val_acc{k_i+1}.png'))

        trainset.close()
        valset.close()
        testset.close()
        del train_imgs, train_labels, val_imgs, val_labels


    valid_acc_avg = np.mean(valid_acc_ks)
    valid_acc_std = np.std(valid_acc_ks)
    print('\nTrain finished.')
    print(f'{N_FOLD+1}-fold Validation accuracy: ', valid_acc_ks)
    print('%d-fold Validation accuracy mean and std: %.4f \u00B1 %.4f' % (N_FOLD+1, valid_acc_avg, valid_acc_std))






