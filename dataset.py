from utils import *

'''
Dataset for segmentation
'''
class AFDataset(Dataset):
    def __init__(self, type, augmentation, aug_list=None, fid=None):
        # open the h5 file
        assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'
        self.type = type
        if type =='test':
            h5_file_name = join(SEG_H5_FILE_PATH, f'{self.type}{fid}.h5')
        else:
            h5_file_name = join(SEG_H5_FILE_PATH, f'{self.type}.h5')
        self.hdf5 = h5py.File(h5_file_name, 'r')
        self.augmentation = augmentation
        self.aug_list = aug_list
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
                img, mask = self.augment(img, mask, self.aug_list) 

        else:
            filename = self.file_keys[2*index]
            
            img = self.hdf5[filename][0]
            mask = self.hdf5[filename][1]
            mask_origin = self.hdf5[filename+'_origin'][:,:,-1]
            # # test
            # if filename == '1564100_OD_2016':
            #     mask_show = mask*255
            #     mask_origin_show = mask_origin*255
            #     cv2.imshow('mask_show', mask_show)
            #     cv2.imshow('mask_origin_show', mask_origin_show)
            #     print()

        
        # Transform to tensor
        img = TF.to_tensor(img)
        mask = torch.tensor(np.array(mask)[:,:,-1], dtype=torch.long)
        # Normalisation
        normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = normalise(img)

        return img, mask, mask_origin, filename

    def augment(self, img, mask, aug_list):
        # transform to pillow image
        img = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)

        if 'crop_rand' in aug_list:
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 256))
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        elif 'crop_stride' in aug_list:
            randid = np.random.randint(25)
            i = 128*(randid//5)
            j = 128*(randid%5)
            img = TF.crop(img, i, j, 256, 256)
            mask = TF.crop(mask, i, j, 256, 256)


        if 'flip' in aug_list:
            # random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # random vertical flipping
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        if 'rotate' in aug_list:
            # random rotation
            deg = transforms.RandomRotation.get_params([-5,5])
            img = TF.rotate(img, deg)
            mask = TF.rotate(mask, deg)

        return img, mask

    def close(self):
        self.hdf5.close()




'''
Dataset for classification
'''
class ASDataset(Dataset):
    def __init__(self, type, transform, fid=None):
        # open the h5 file
        assert type == 'train' or 'val' or 'test', 'Unrecognizable dataset type'

        if type =='test':
            h5_file_name = join(CLA_H5_FILE_PATH, f'{type}{fid}.h5')
        else:
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


