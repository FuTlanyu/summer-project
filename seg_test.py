# import os
# from utils import *
# from os.path import join
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import torch
# import torchvision.transforms as transforms
# from torch import optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.utils import shuffle
# from SegNet import *
# from tqdm import tqdm
# from sklearn.model_selection import KFold
# import h5py
# import time
# from PIL import Image



# if __name__ == '__main__':

#     BATCHSIZE_TE = 1

#     transform_test = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         # print("Device: ", torch.cuda.get_device_name(device))
#         torch.cuda.empty_cache()
    
#     criterion = torch.nn.CrossEntropyLoss()

#     testset = AFDataset('test', transform_test)
#     testloader = DataLoader(testset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)

#     ## load the trained model
#     segnet = SegNet()
#     segnet.load_state_dict(torch.load('model/class_model_lite1.pt'))

#     test_img = Image.open(r'C:\Users\Fu Tianyu\Documents\UCL\CSML\Final Project\CSML_Summer2022\Data\region_of_interest\data_raw\USH\221131_OD_2013.tif')
#     test_img = transform_test(test_img)
#     test_img = test_img[None]
#     output = segnet(test_img)
#     pred = output.argmax(dim=1)
#     display_image(pred, 'test_pred.png')
#     print('finished')


#     # test_loss = 0
#     # mIoU_sum_test = 0
#     # total_test = 0
#     # for batch_idx, (inputs, targets, targets_origin) in enumerate(testloader):

#     #     inputs, targets = inputs.to(device), targets.to(device)
#     #     outputs = segnet(inputs)
#     #     loss = criterion(outputs, targets)

#     #     test_loss += loss.item()
#     #     preds = outputs.argmax(dim=1)
#     #     # restore to original mask image size then compute miou 
#     #     preds_restore = image_restore(preds, targets_origin.shape)


#     #     display_image(preds, f'{k_i}fold_test_preds{batch_idx+1}.png')
#     #     display_image(preds_restore, f'{k_i}fold_test_preds_restore{batch_idx+1}.png')
#     #     display_image(targets_origin, f'{k_i}fold_test_gt{batch_idx+1}.png')

#     #     mIoU_sum_test += mIoU_score(preds_restore, targets_origin)
#     #     total_test += targets.size(0)





    
#     # outputs = segnet(images)
#     # _, predicted = torch.max(outputs, 1)
#     # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#     # # save to images
#     # im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
#     # im.save("test_pt_images.jpg")
#     # print('test_pt_images.jpg saved.')