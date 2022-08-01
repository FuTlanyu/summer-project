from utils import *
from mtl_model import MTL
from ClaNet import ClaNet
from SegNet import SegNet
from dataset import AFDataset, ASDataset



def dev_test_split_cla():
    img_path_all = []
    label_all = []
    for root, _, files in os.walk(CLA_DATA_NEW_PATH):
        if len(files) != 0:
            for file in files:
                img_path_all.append(os.path.join(root, file))
                if 'AF' in file:
                    label_all.append(1)
                elif 'SLO' in file:
                    label_all.append(0)
 
    img_path_all, label_all = shuffle(img_path_all, label_all, random_state=39)
    assert len(img_path_all) == len(label_all), 'cla image and label size does not match'
    N = len(img_path_all)
    
    # split whole dataset into training and test set
    test_ratio = 0.2
    split_idx = int(N*test_ratio)
    test_imgs = img_path_all[:split_idx]
    test_labels = label_all[:split_idx]
    print(f'Cla test size {len(test_imgs)}')

    dev_imgs = img_path_all[split_idx:]
    dev_labels = label_all[split_idx:]

    return test_imgs, test_labels, dev_imgs, dev_labels


def dev_test_split_seg():
    img_paths = []
    mask_paths = []
    for root, _, files in os.walk(SEG_DATA_PATH):
        if len(files) != 0:
            for file in files:
                if '.tif' in file: 
                    img_paths.append(os.path.join(root, file))
                elif '.jpg' or '.gif' in file:
                    mask_paths.append(os.path.join(root, file))
    
    img_paths = sorted(img_paths)
    mask_paths = sorted(mask_paths)
    img_paths, mask_paths = shuffle(img_paths, mask_paths, random_state=39)
    assert len(img_paths) == len(mask_paths), 'seg image and mask number don\'t match'
    N = len(img_paths)

    for i in range(len(img_paths)):
        img_name = os.path.split(img_paths[i])[-1].split('.')[0]
        mask_name = os.path.split(mask_paths[i])[-1].split('.')[0]
        assert img_name == mask_name, "seg image and mask don't match"

    ### Split dataset into training and test set
    test_ratio = 0.2
    split_idx = int(N*test_ratio)
    test_imgs = img_paths[:split_idx]
    test_masks = mask_paths[:split_idx]
    print(f'Seg test size {len(test_imgs)}')

    dev_imgs = img_paths[split_idx:]
    dev_masks = mask_paths[split_idx:]

    return test_imgs, test_masks, dev_imgs, dev_masks 




def cs_weight_statics(model_param):
    """Compute the cross stitch weights mean and variance over channels for each cross-stitch unit

    Args:
        model_param: named model parameters
    """
    statics = []
    for name, param in model_param:
        if param.requires_grad and 'cross_stitch' in name:
            cs_weights = param.cpu().detach().numpy()
            chns = cs_weights.shape[0]
            p_chns = [[cs_weights[chn,0,0]+cs_weights[chn,1,1], cs_weights[chn,0,1]+cs_weights[chn,1,0]] for chn in range(chns)] # (chns, 2)
            p_chns = np.array(p_chns)/2
            p_same_mean = np.nanmean(p_chns[:,0])
            p_same_std = np.nanstd(p_chns[:,0])
            p_diff_mean = np.nanmean(p_chns[:,1])
            p_diff_std = np.nanstd(p_chns[:,1])

            statics.append(p_same_mean)
            statics.append(p_same_std)
            statics.append(p_diff_mean)
            statics.append(p_diff_std)
    
    assert len(statics) == 20
    return np.array(statics)


def cla_pretrain(trainset, valset, batchsize, lr, epochs, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    mtl_clanet = ClaNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(mtl_clanet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    validloader = DataLoader(valset, batch_size=batchsize, shuffle=True, num_workers=0)
    # num_batch_train = len(trainloader)
    # num_batch_val = len(validloader)


    print('\nClassfication parameters pre-training...')
    # save the train/valid loss/metric for every epoch 
    # history = torch.zeros((EPOCH, 4))
    for epoch in range(epochs):
        ################### Training ###################
        mtl_clanet.train()
        # accumulate loss and accuarcy over batches
        train_loss = 0
        correct = 0
        total = 0
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Train Epoch {epoch+1}")
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = mtl_clanet(inputs)
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
        # history[epoch][0] = train_loss / num_batch_train
        # history[epoch][1] = correct / total

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ################### Validation ################### 
        mtl_clanet.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            with tqdm(validloader, unit="batch") as vepoch:
                for batch_idx, (inputs, targets) in enumerate(vepoch):
                    vepoch.set_description(f"Valid Epoch {epoch+1}")

                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = mtl_clanet(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    total_val += targets.size(0)
                    correct_val += preds.eq(targets).sum().item()
            
                    vepoch.set_postfix(loss=val_loss/(batch_idx+1), accuracy=100.*correct_val/total_val)

        acc = 100.*correct_val/total_val
        # print(f'valid epoch {epoch} total {total_val}')
        
        # history[epoch][2] = val_loss / num_batch_val
        # history[epoch][3] = correct_val / total_val
        
        # print('Training: loss: %.5f, accuracy: %.5f' % (history[epoch][0], history[epoch][1]))
        # print('Validation: loss: %.5f, accuracy: %.5f' % (history[epoch][2], history[epoch][3]))
    return mtl_clanet.state_dict()


def seg_pretrain(trainset, valset, batchsize, lr, epochs, device):
    if torch.cuda.is_available():
        # print("Device: ", torch.cuda.get_device_name(device))
        torch.cuda.empty_cache()
    
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
    validloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    mtl_segnet = SegNet(bayes=False, dropout=False).to(device)
    criterion = DiceBCELoss()

    optimizer = optim.SGD(mtl_segnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # num_batch_train = len(trainloader)
    # num_batch_val = len(validloader)

    print('\nSegmentation parameters pre-training...')
    # save the train/valid loss/metric downsampled valid metric for every epoch 
    # history = torch.zeros((EPOCH, 5))
    for epoch in range(epochs):
        ################### Training ###################
        mtl_segnet.train()
        # accumulate loss and mIoU over batches
        train_loss = 0
        mIoU_sum = 0
        total = 0
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Train Epoch {epoch+1}")
            for batch_idx, (inputs, targets, _, _) in enumerate(tepoch):
                
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = mtl_segnet(inputs)
                loss = criterion(outputs, targets)
            
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                mIoU_sum += mIoU_score(preds, targets)
                total += targets.size(0)

                tepoch.set_postfix(loss=train_loss/(batch_idx+1), mIoU=100.*mIoU_sum/total)
        
        # history[epoch][0] = train_loss / num_batch_train
        # history[epoch][1] = mIoU_sum / total
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ################### Validation ################### 
        mtl_segnet.eval()

        val_loss = 0
        mIoU_sum_val = 0
        total_val = 0
        with torch.no_grad():
            with tqdm(validloader, unit="batch") as vepoch:
                for batch_idx, (inputs, targets, targets_origins, filenames) in enumerate(vepoch):
                    vepoch.set_description(f"Valid Epoch {epoch+1}")
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = mtl_segnet(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)

                    # restore to original mask image size then compute miou 
                    preds_restore = image_restore_seg(preds, targets_origins.shape)
                    mscore = mIoU_score(preds_restore, targets_origins)
                    mIoU_sum_val += mscore
                    total_val += targets.size(0)

                    vepoch.set_postfix(loss=val_loss/(batch_idx+1), mIoU=100.*mIoU_sum_val/total_val)
        
        # history[epoch][2] = val_loss / num_batch_val
        # history[epoch][3] = mIoU_sum_val / total_val
        # history[epoch][4] = mIoU_sum_val_ds / total_val

    return mtl_segnet.state_dict()




if __name__ == '__main__':

    BATCHSIZE_TR = 1
    BATCHSIZE_TE_SEG = 1
    LR = 0.001

    NORM_SIZE = 512
    aug_train_bool = False
    aug_list = ['rotate', 'flip']
    clahe_bool = True
    
    p_same = 0.8
    p_diff = 0.2
    BATCHSIZE_TE_CLA = 1
    # report the result every BATCH_PER_EPOCH batches
    BATCH_PER_EPOCH = 1 #10
    EPOCH = 2 #60
    compute_wcs = False
    
    cs_diff_lr = True
    CS_LR_RATIO = 100

    pretrain_load = False
    PRE_SEG_EPOCH = 100
    PRE_CLA_EPOCH = 15

    print('Loading data...')
    ## Test and development set split
    seg_test_imgs, seg_test_masks, seg_dev_imgs, seg_dev_masks = dev_test_split_seg()
    cla_test_imgs, cla_test_labels, cla_dev_imgs, cla_dev_labels = dev_test_split_cla()
    
    ## Save test set as HDF5
    create_h5_cla(cla_test_imgs, cla_test_labels, 'test')
    create_h5_test_seg(seg_test_imgs, seg_test_masks, 'test')


    cla_transform_train = transforms.Compose(
        [transforms.ToPILImage(),
        # transforms.RandomCrop(512),
        # transforms.RandomRotation((0,180)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cla_transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    ### N-fold split
    N_FOLD = 5
    kf = KFold(n_splits=N_FOLD)
    seg_kf_gen = kf.split(seg_dev_imgs)
    cla_kf_gen = kf.split(cla_dev_imgs)

    # miou of the last epoch
    val_miou_per_fold = np.zeros(N_FOLD)
    train_miou_per_fold = np.zeros(N_FOLD)
    # miou of the best epoch
    val_miou_best_per_fold = np.zeros(N_FOLD)
    best_model_paths = []
    for k_i in range(N_FOLD):
        seg_train_index, seg_val_index = next(seg_kf_gen)
        cla_train_index, cla_val_index = next(cla_kf_gen)
        
        print(f'\n{k_i+1} fold multi-task model')
        print(f'Seg train size: {len(seg_train_index)} val size: {len(seg_val_index)}')
        print(f'Cla train size: {len(cla_train_index)} val size: {len(cla_val_index)}')
        
        seg_train_imgs, seg_train_masks = data_extract(seg_train_index, seg_dev_imgs, seg_dev_masks)
        seg_val_imgs, seg_val_masks = data_extract(seg_val_index, seg_dev_imgs, seg_dev_masks)

        cla_train_imgs, cla_train_labels = data_extract(cla_train_index, cla_dev_imgs, cla_dev_labels)
        cla_val_imgs, cla_val_labels = data_extract(cla_val_index, cla_dev_imgs, cla_dev_labels)
        


        # create hdf5 file for each split
        print('\nGenerating HDF5 file...')
        t1 = time.time()
        create_h5_train_seg(seg_train_imgs, seg_train_masks, 'train')
        create_h5_test_seg(seg_val_imgs, seg_val_masks, 'val')
        create_h5_cla(cla_train_imgs, cla_train_labels, 'train')
        create_h5_cla(cla_val_imgs, cla_val_labels, 'val')
        t2 = time.time()
        print('HDF5 file generation time: %.2f (min)' % ((t2-t1)/60))


        seg_trainset = AFDataset('train', augmentation=aug_train_bool)
        seg_valset = AFDataset('val', augmentation=False)
        seg_testset = AFDataset('test', augmentation=False)

        cla_trainset = ASDataset('train', cla_transform_train)
        cla_valset = ASDataset('val', cla_transform_test)
        cla_testset = ASDataset('test', cla_transform_test)

        seg_trainloader = DataLoader(seg_trainset, batch_size=BATCHSIZE_TR, shuffle=True, drop_last=True, num_workers=0)
        seg_validloader = DataLoader(seg_valset, batch_size=BATCHSIZE_TE_SEG, shuffle=False, num_workers=0)
        seg_testloader = DataLoader(seg_testset, batch_size=BATCHSIZE_TE_SEG, shuffle=False, num_workers=0)

        cla_trainloader = DataLoader(cla_trainset, batch_size=BATCHSIZE_TR, shuffle=True, drop_last=True, num_workers=0)
        cla_validloader = DataLoader(cla_valset, batch_size=BATCHSIZE_TE_CLA, shuffle=True, num_workers=0)
        cla_testloader = DataLoader(cla_testset, batch_size=BATCHSIZE_TE_CLA, shuffle=True, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # print("Device: ", torch.cuda.get_device_name(device))
            torch.cuda.empty_cache()
        
        mtl = MTL(p_same=p_same, p_diff=p_diff, device=device).to(device)

        if pretrain_load:
            seg_state_dict = seg_pretrain(seg_trainset, seg_valset, batchsize=16, lr=0.001, epochs=PRE_SEG_EPOCH, device=device)
            cla_state_dict = cla_pretrain(cla_trainset, cla_valset, batchsize=16, lr=0.001, epochs=PRE_CLA_EPOCH, device=device)
            seg_state_dict = change_state_keys(seg_state_dict, type='seg')
            cla_state_dict = change_state_keys(cla_state_dict, type='cla')
            
            mtl.load_state_dict(seg_state_dict, strict=False)
            mtl.load_state_dict(cla_state_dict, strict=False)
            mtl_state_dict = mtl.state_dict()
        
        
        criterion = MTLLoss()
        if not cs_diff_lr:
            optimizer = optim.SGD(mtl.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        else:
            wcs_list = ['cross_stitch1.weights','cross_stitch2.weights','cross_stitch3.weights','cross_stitch4.weights','cross_stitch5.weights']
            cs_params = list(filter(lambda kv: kv[0] in wcs_list, mtl.named_parameters()))
            cs_params = list(map(lambda x: x[1], cs_params))
            base_params = list(filter(lambda kv: kv[0] not in wcs_list, mtl.named_parameters()))
            base_params = list(map(lambda x: x[1], base_params))
            
            # optimizer = optim.SGD([
            #     {'params': cs_params, 'lr': LR*CS_LR_RATIO},
            #     {'params': base_params, 'lr': LR}
            # ], lr=LR, momentum=0.9, weight_decay=5e-4)
            optimizer = optim.SGD([
                {'params': cs_params, 'lr': LR*CS_LR_RATIO, 'momentum': 0.9, 'weight_decay': 5e-4},
                {'params': base_params, 'lr': LR, 'momentum': 0.9, 'weight_decay': 5e-4}])

        seg_train_iter = iter(seg_trainloader)
        cla_train_iter = iter(cla_trainloader)
        seg_val_num_batch = len(seg_validloader)
        
        print('\nTraining...')
        # cs layer 1-5 mean and std of p_same/p_diff
        cs_param_history = np.zeros((EPOCH, 20))
        # save the train loss/mIoU/acc val loss/mIoU  for every epoch
        history = torch.zeros((EPOCH, 5))
        val_miou_best = 0.0
        model_best_path = None
        for epoch in range(EPOCH):
            ################### Training ###################
            mtl.train()
            train_loss = 0
            cla_correct = 0
            mIoU_sum = 0
            total = 0
            
            for batch_id in range(BATCH_PER_EPOCH):
                # get a batch of seg/cla data
                try:
                    seg_inputs, seg_targets, _, _ = next(seg_train_iter)
                except StopIteration:
                    seg_train_iter = iter(seg_trainloader)
                    seg_inputs, seg_targets, _, _ = next(seg_train_iter)

                try:
                    cla_inputs, cla_targets = next(cla_train_iter)
                except StopIteration:
                    cla_train_iter = iter(cla_trainloader)
                    cla_inputs, cla_targets = next(cla_train_iter)
                
                # forward and backward
                seg_inputs, cla_inputs = seg_inputs.to(device), cla_inputs.to(device)  
                seg_targets, cla_targets = seg_targets.to(device), cla_targets.to(device)  
                optimizer.zero_grad()

                seg_outputs, cla_outputs = mtl(seg_inputs, cla_inputs)
                loss = criterion(seg_outputs, cla_outputs, seg_targets, cla_targets)
            
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                seg_preds = seg_outputs.argmax(dim=1)
                mIoU_sum += mIoU_score(seg_preds, seg_targets)
                
                _, cla_preds = cla_outputs.max(1)
                cla_correct += cla_preds.eq(cla_targets).sum().item()
                
                total += seg_targets.size(0)

            history[epoch, 0] = train_loss / BATCH_PER_EPOCH
            history[epoch, 1] = cla_correct / total
            history[epoch, 2] = mIoU_sum / total 

            print('Epoch %d train loss: %.4f acc: %.4f mIoU: %.4f' % (epoch+1, history[epoch, 0], history[epoch, 1], history[epoch, 2]))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ################### Validation ################### 
            if compute_wcs:
                cs_param_history[epoch, :] = cs_weight_statics(mtl.named_parameters())
                # print(cs_param_history[epoch, :].reshape((5,4)))
            
            mtl.eval()
            # test with dropout
            # if bayes_bool:
            #     mtl.apply(apply_dropout)
            # t1 = time.time()
            loss_val = 0
            mIoU_sum_val = 0
            total_val = 0
            with torch.no_grad():
                cla_val_iter = iter(cla_validloader)
                for batch_idx, (seg_inputs, seg_targets, seg_targets_origins, seg_filenames) in enumerate(seg_validloader):
                    
                    cla_inputs, cla_targets = next(cla_val_iter)
                    seg_inputs, cla_inputs = seg_inputs.to(device), cla_inputs.to(device)
                    seg_targets, cla_targets = seg_targets.to(device), cla_targets.to(device) 
                    # 1 seg to 20 clas making up of 20 (seg, cla) pairs

                    var_maps = [None]
                    if BATCHSIZE_TE_CLA == 1:
                        seg_outputs, cla_outputs = mtl(seg_inputs, cla_inputs)
                        loss = criterion(seg_outputs, cla_outputs, seg_targets, cla_targets)
                        loss_val += loss.item()
                        seg_preds = seg_outputs.argmax(dim=1)
                    else:
                        loss_match_total = []
                        probs_total = []
                        for run in range(BATCHSIZE_TE_CLA):
                            cla_inputs_one = cla_inputs[run][None,:,:,:]
                            cla_targets_one = cla_targets[run][None]
                            seg_outputs, cla_outputs = mtl(seg_inputs, cla_inputs_one)
                    
                            loss = criterion(seg_outputs, cla_outputs, seg_targets, cla_targets_one)
                            loss_match_total.append(loss.item())

                            probs = F.softmax(seg_outputs, dim=1)
                            probs_total.append(probs.cpu().detach().numpy())

                        probs_mean = np.nanmean(probs_total, axis=0)
                        probs_var = np.var(probs_total, axis=0)

                        loss_val += np.nanmean(loss_match_total)
                        seg_preds = probs_mean.argmax(axis=1)

                        # generate the uncertainty map (probs_var, preds)
                        var_maps = np.take_along_axis(probs_var, seg_preds[:,None],axis=1)[:,0]

                    # restore to original mask image size then compute miou 
                    seg_preds_restore = image_restore_seg(seg_preds, seg_targets_origins.shape)
                                            
                    mscore_re = mIoU_score(seg_preds_restore, seg_targets_origins)
                    mscore = mIoU_score(seg_preds, seg_targets)

                    if epoch == EPOCH-1:
                        # seg_visual_mtl(seg_preds_restore[0], var_maps[0], mscore_re, seg_filenames[0], f'fold{k_i+1}_val_pred{batch_idx+1}.png')
                        seg_visual_overlap(seg_targets[0], seg_preds[0], var_maps[0], seg_filenames[0], 'fold%d_val_pred%d_miou%.3f_%s.png'%(k_i+1,batch_idx+1,mscore,seg_filenames[0]))

                    mIoU_sum_val += mscore
                    total_val += seg_targets.size(0)

            history[epoch, 3] = loss_val / seg_val_num_batch
            history[epoch, 4] = mIoU_sum_val / total_val

            print('Epoch %d valid loss: %.4f mIoU: %.4f' % (epoch+1, history[epoch, 3], history[epoch, 4]))
            # Save the model(epoch) with highest val miou during training
            if history[epoch][4] > val_miou_best:
                if model_best_path is not None:
                    os.remove(model_best_path)
                modelpath = join(MTL_MODEL_PATH, 'fold%d_epoch%d_%.3f_lite.pt' % (k_i+1, epoch+1, history[epoch][4]))
                torch.save(mtl.state_dict(), modelpath)
                val_miou_best = history[epoch][4]
                model_best_path = modelpath
            
            # t2 = time.time()
            # print('Validation time: %.2f (min)' % ((t2-t1)/60))

        train_miou_per_fold[k_i] = history[-1,2]
        val_miou_per_fold[k_i] = history[-1,-1]
        val_miou_best_per_fold[k_i] = val_miou_best
        best_model_paths.append(split(model_best_path)[-1])

        
        
        ### plot training and validation history
        x = np.arange(EPOCH)
        plt.figure()
        plt.plot(x, history[:,0], label='train loss') # train loss
        plt.plot(x, history[:,3], label='val loss') # val loss
        plt.legend()
        plt.title('Training and validation loss for {} epochs'.format(EPOCH))
        plt.savefig(join(MTL_MODEL_PATH, f'train_val_loss{k_i+1}.png'))

        plt.figure()
        plt.plot(x, history[:,2], label='train mIoU') 
        plt.plot(x, history[:,4], label='val mIoU') 
        plt.legend()
        plt.title('Training and validation mIoU for {} epochs'.format(EPOCH))
        plt.savefig(join(MTL_MODEL_PATH, f'train_val_miou{k_i+1}.png'))

        plt.figure()
        plt.plot(x, history[:,1], label='train cla acc') 
        plt.legend()
        plt.title('Training classification accuracy for {} epochs'.format(EPOCH))
        plt.savefig(join(MTL_MODEL_PATH, f'train_cla_acc{k_i+1}.png'))

        # plot the p_same and p_diff changing curves
        # 1-4 p_same_mean, p_same_std, p_diff_mean, p_diff_std
        if compute_wcs:
            ## mean
            plt.figure()
            for chn in range(5):
                plt.plot(x, cs_param_history[:,4*chn], label=f'unit{chn+1} p same mean')
                plt.plot(x, cs_param_history[:,4*chn+2], label=f'unit{chn+1} p diff mean')
            plt.legend()
            plt.title(f'Cross stitch weights mean for {EPOCH} epochs')
            plt.savefig(join(MTL_MODEL_PATH, f'cs_weights_mean{k_i+1}.png'))

            ## std
            plt.figure()
            for chn in range(5):
                plt.plot(x, cs_param_history[:,4*chn+1], label=f'unit{chn+1} p same std')
                plt.plot(x, cs_param_history[:,4*chn+3], label=f'unit{chn+1} p diff std')
            plt.legend()
            plt.title(f'Cross stitch weights std for {EPOCH} epochs')
            plt.savefig(join(MTL_MODEL_PATH, f'cs_weights_std{k_i+1}.png'))
        
        plt.close('all')
        seg_trainset.close()
        seg_valset.close()
        seg_testset.close()
        cla_trainset.close()
        cla_valset.close()
        cla_testset.close()
        
    train_miou_avg = np.mean(train_miou_per_fold)
    train_miou_std = np.std(train_miou_per_fold)
    val_miou_avg = np.mean(val_miou_per_fold)
    val_miou_std = np.std(val_miou_per_fold)
    val_miou_best_avg = np.mean(val_miou_best_per_fold)
    val_miou_best_std = np.std(val_miou_best_per_fold)

    print('\nTrain finished.')
    print(f'{N_FOLD}-fold train mIoU: ', train_miou_per_fold)
    print('%d-fold train mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, train_miou_avg, train_miou_std))
    
    print(f'\n{N_FOLD}-fold validation mIoU: ', val_miou_per_fold)
    print('%d-fold validation mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, val_miou_avg, val_miou_std))

    print(f'\n{N_FOLD}-fold optimal validation mIoU: ', val_miou_best_per_fold)
    print('%d-fold optimal validation mIoU mean and std: %.4f \u00B1 %.4f' % (N_FOLD, val_miou_best_avg, val_miou_best_std))
    print('Best models: ', best_model_paths)
    