import os

from cv2 import sort
from cvxpy import length

from utils import *
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from SegNet import *
from tqdm import tqdm
from sklearn.model_selection import KFold
import h5py
import time



'''
transform images or masks into an array of images with uniform image size
'''
def load_imgs(data_paths, data_type, transform=True):
    # img_paths = list(img_paths)
    images = []

    for img_path in data_paths:
        img = cv2.imread(img_path)
        if data_type == 'mask':
            img = img/255
        
        if transform:
            # TODO crop and resize image to the same size
            norm_size = 256
            x, y, _ = img.shape
            if x > y:
                img = img[:y,:]
            if x < y:
                img = img[:,:x]
            assert img.shape[0] == img.shape[1]
            if img.shape[0] != norm_size:
                img = cv2.resize(img, (norm_size, norm_size))
        
        images.append(np.array(img))

    return np.array(images, dtype='uint8')



def create_h5_train(imgs, masks, type):
    """Create the hdf5 file for each subset containing images and masks

    Args:
        imgs: a list of train image paths
        masks: a list of train mask paths
    """
    trans = True
    imgs_arr = load_imgs(imgs, 'image', transform=trans)
    masks_arr = load_imgs(masks, 'mask', transform=trans)

    h5_file_name = join(H5_FILE_PATH, f'{type}.h5')
    with h5py.File(h5_file_name, 'w') as f:
        f.create_dataset(name='imgs', data=imgs_arr)
        f.create_dataset(name='masks', data=masks_arr)


def create_h5_test(imgs, masks, type):
    """Create the hdf5 file for each subset

    Args:
        imgs: a list of val/test image paths
        masks: a list of val/test mask paths
        type: 'val' or 'test'
    """

    h5_file_name = join(H5_FILE_PATH, f'{type}.h5')
    with h5py.File(h5_file_name, 'w') as f:
        for img, mask in zip(imgs, masks):
            img_name = os.path.split(img)[-1].split('.')[0]
            mask_name = os.path.split(mask)[-1].split('.')[0]
            assert img_name == mask_name, "image and mask don't match"
            
            img_arr = cv2.imread(img)
            mask_arr = cv2.imread(mask)/255

            f.create_dataset(name=img_name, data=np.array([img_arr, mask_arr], dtype='uint8'))



def seg_visual(pred_mask, img_name, fig_name):
    """Display image, ground truth, predicted mask

    Args:
        pred_mask: predicted mask
        img_name: image filename
    """
    if type(pred_mask) != np.ndarray:
        pred_mask = pred_mask.cpu().detach().numpy()
    fig, axs = plt.subplots(1, 3)
    
    if 'RPGR' in img_name:
        img_path = join(RPGR_PATH, img_name+'.tif')
        gt_path = join(RPGR_PATH, img_name+'.jpg')
    else:
        img_path = join(USH_PATH, img_name+'.tif')
        gt_path = join(USH_PATH, img_name+'.jpg')
    
    img = plt.imread(img_path)
    gt = plt.imread(gt_path)
    pred_mask = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)*255
    
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[1].imshow(gt)
    axs[1].axis('off')
    axs[2].imshow(pred_mask)
    axs[2].axis('off')
    
    # fig.suptitle('')
    plt.savefig(join(VISUAL_PATH, fig_name))  
    plt.close()


'''
Compute the sum of the mIoU scores of a batch of data
'''
def mIoU_score(preds, targets, num_classes=2):
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



class AFDataset(Dataset):
    def __init__(self, type, transform):
        # open the h5 file
        assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'
        self.type = type
        h5_file_name = join(H5_FILE_PATH, f'{self.type}.h5')
        self.hdf5 = h5py.File(h5_file_name, 'r')
        self.transform = transform
        if self.type != 'train':
            self.filenames = list(self.hdf5.keys())
            print(f'{self.type} dataset: {self.filenames}')

    def __len__(self):
        if self.type == 'train':
            self.length = self.hdf5['masks'].shape[0]
        else: 
            self.length = len(self.filenames)
        
        return self.length
    
    def __getitem__(self, index):
        filename = ''
        if self.type == 'train':        
            img = self.hdf5['imgs'][index]
            mask = self.hdf5['masks'][index]        
        else:
            # original test / val image and mask
            filename = self.filenames[index]
            img = self.hdf5[filename][0]
            mask = self.hdf5[filename][1]
            
        img = self.transform(img)
        mask = torch.tensor(mask[:,:,-1], dtype=torch.long)

        return img, mask, filename

    def close(self):
        self.hdf5.close()











if __name__ == '__main__':

    BATCHSIZE_TR = 4
    BATCHSIZE_TE = 1
    LR = 0.001
    EPOCH = 100

    print('Loading data...')
    img_paths = []
    mask_paths = []
    for root, dirs, files in os.walk(DATA_PATH):
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
    test_ratio = 0.2
    split_idx = int(N*test_ratio)
    test_imgs = img_paths[:split_idx]
    test_masks = mask_paths[:split_idx]
    print(f'test size {len(test_imgs)}')

    create_h5_test(test_imgs, test_masks, 'test')

    trv_imgs = img_paths[split_idx:]
    trv_masks = mask_paths[split_idx:]

    
    transform_train = transforms.Compose(
        [transforms.ToPILImage(),

        # transforms.RandomRotation((0,180)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    N_FOLD = 5
    valid_miou_per_fold = np.zeros(N_FOLD)

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
        create_h5_train(train_imgs, train_masks, 'train')
        create_h5_test(val_imgs, val_masks, 'val')
        t2 = time.time()
        print('hdf5 file generation time: %.2f (min)' % ((t2-t1)/60))

        trainset = AFDataset('train', transform_train)
        valset = AFDataset('val', transform_test)
        testset = AFDataset('test', transform_test)

        trainloader = DataLoader(trainset, batch_size=BATCHSIZE_TR, shuffle=True, num_workers=0)
        validloader = DataLoader(valset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # print("Device: ", torch.cuda.get_device_name(device))
            torch.cuda.empty_cache()
        
        segnet = SegNet().to(device)
        

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(segnet.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

        num_batch_train = len(trainloader)
        num_batch_val = len(validloader)
        num_batch_test = len(testloader)


        print('\nTraining...')
        # save the train/valid loss/metric for every epoch 
        history = torch.zeros((EPOCH, 4))
        for epoch in range(EPOCH):
            ################### Training ###################
            segnet.train()
            # accumulate loss and mIoU over batches
            train_loss = 0
            mIoU_sum = 0
            total = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Train Epoch {epoch+1}")
                for batch_idx, (inputs, targets, _) in enumerate(tepoch):
                    
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
            val_loss = 0
            mIoU_sum_val = 0
            total_val = 0
            with torch.no_grad():
                with tqdm(validloader, unit="batch") as vepoch:
                    for batch_idx, (inputs, targets, filenames) in enumerate(vepoch):
                        vepoch.set_description(f"Valid Epoch {epoch+1}")

                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = segnet(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        
                        if epoch == EPOCH-1 and k_i == N_FOLD-1:
                            seg_visual(preds[0], filenames[0], f'val_pred{batch_idx+1}.png')

                        mIoU_sum_val += mIoU_score(preds, targets)
                        total_val += targets.size(0)
                
                        vepoch.set_postfix(loss=val_loss/(batch_idx+1), mIoU=100.*mIoU_sum_val/total_val)
            
            history[epoch][2] = val_loss / num_batch_val
            history[epoch][3] = mIoU_sum_val / total_val
            
            # print('Training: loss: %.5f, mIoU: %.5f' % (history[epoch][0], history[epoch][1]))
            # print('Validation: loss: %.5f, mIoU: %.5f' % (history[epoch][2], history[epoch][3]))
        
        valid_miou_per_fold[k_i] = history[-1,-1]
        torch.save(history, join(MODEL_PATH, f'history{k_i+1}.pt'))
        torch.save(segnet.state_dict(), join(MODEL_PATH, f'class_model_lite{k_i+1}.pt'))


        ### plot training and validation history
        x = np.arange(EPOCH)
        plt.figure()
        plt.plot(x, history[:,0], label='train loss') # train loss
        plt.plot(x, history[:,2], label='val loss') # val loss
        plt.legend()
        plt.title('Training and validation loss for {} epochs'.format(EPOCH))
        plt.savefig(join(MODEL_PATH, f'train_val_loss{k_i+1}.png'))

        plt.figure()
        plt.plot(x, history[:,1], label='train mIoU') 
        plt.plot(x, history[:,3], label='val mIoU') 
        plt.legend()
        plt.title('Training and validation mIoU for {} epochs'.format(EPOCH))
        plt.savefig(join(MODEL_PATH, f'train_val_miou{k_i+1}.png'))

        trainset.close()
        valset.close()
        testset.close()
        del train_imgs, train_masks, val_imgs, val_masks


    valid_miou_avg = np.mean(valid_miou_per_fold)
    valid_miou_std = np.std(valid_miou_per_fold)
    print('\nTrain finished.')
    print(f'{N_FOLD}-fold Validation mIoU: ', valid_miou_per_fold)
    print('%d-fold Validation mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD+1, valid_miou_avg, valid_miou_std))



