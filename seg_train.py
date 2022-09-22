from os import kill
from utils import *
from SegNet import SegNet
from UNet import UNet
from dataset import AFDataset


def seg_inference(post_process_bool, thresh, model, criterion, dataloader, norm_size, num_batch, kid, bayes_bool, num_runs_bayes, device, infer_type, last_epoch=False):
    assert infer_type == 'test' or 'val', 'No inference type'
    loss_infer = 0
    # mIoU_sum_val_ds = 0
    mIoU_sum = 0
    total = 0
    uncertainty_sum = 0
    y_gt_all = [] 
    y_prob_all = [] 

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as vepoch:
            for batch_idx, (inputs, targets, targets_origins, filenames) in enumerate(vepoch):
                if infer_type == 'test':
                    vepoch.set_description(f"Test Model {kid+1}")
                else:
                    vepoch.set_description(f"Valid Epoch {epoch+1}")
                
                inputs, targets = inputs.to(device), targets.to(device)
                var_maps = [None]
                
                if bayes_bool:
                    # model uncertainty analysis
                    loss_bayes_total = []
                    probs_total = []
                    for run in range(num_runs_bayes):
                        if norm_size != 512:
                            outputs = patch_infer_output(model, inputs, norm_size, device)
                        else:    
                            outputs = model(inputs)
                
                        loss = criterion(outputs, targets)
                        loss_bayes_total.append(loss.item())

                        probs = F.softmax(outputs, dim=1)
                        probs_total.append(probs.cpu().detach().numpy())

                    probs_mean = np.nanmean(probs_total, axis=0)
                    probs_var = np.var(probs_total, axis=0) # (1,2,512,512)

                    # preds = probs_mean.argmax(axis=1) # (1,512,512)
                    preds = np.where(probs_mean[:,1,:,:]>thresh, 1, 0)

                    # generate the uncertainty map (probs_var, preds)
                    var_maps = np.take_along_axis(probs_var, preds[:,None],axis=1)[:,0] # (1,512,512)
                    uncertainty_sum += np.sum(var_maps)
                    loss_infer += np.nanmean(loss_bayes_total)
                    y_prob_all.append(probs_mean[:,1,:,:])

                else:
                    if norm_size != 512:
                        outputs = patch_infer_output(model, inputs, norm_size, device)
                    else:    
                        outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss_infer += loss.item()
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.where(probs[:,1,:,:]>thresh, 1, 0)
                    y_prob_all.append(probs[:,1,:,:].cpu().detach().numpy())

                y_gt_all.append(targets[0].cpu().detach().numpy())

                if torch.is_tensor(preds[0]):
                    preded = preds[0].cpu().detach().numpy()
                else: preded = preds[0]
                # post process
                if post_process_bool and infer_type == 'test' :
                    preded = post_process(preded, norm_size)

                # restore to original mask image size then compute miou 
                pred_restore = image_restore_seg(preded, targets_origins[0].shape)
                # mscore_ds = mIoU_score(preds, targets)
                mscore = mIoU_score(pred_restore, targets_origins[0])

                if infer_type == 'test' or last_epoch:
                    img_visual_name = 'fold%d_%s_pred%d_miou%.3f_%s.png'%(kid+1,infer_type,batch_idx+1,mscore,filenames[0])
                    seg_visual(pred_restore, var_maps[0], mscore, filenames[0], img_visual_name, 'seg')
                    seg_visual_overlap(norm_size, targets[0], preded, var_maps[0], filenames[0], img_visual_name, 'seg')
                
                # mIoU_sum_val_ds += mscore_ds
                mIoU_sum += mscore
                total += targets.size(0)

                vepoch.set_postfix(loss=loss_infer/(batch_idx+1), mIoU=100.*mIoU_sum/total)
    
    y_prob_all = np.array(y_prob_all)
    y_gt_all = np.array(y_gt_all)
    fpr, tpr, _ = roc_curve(y_gt_all.reshape(-1), y_prob_all.reshape(-1))
    auc_value = auc(fpr, tpr)

    loss_avg = loss_infer/num_batch
    mIoU_avg = mIoU_sum/total
    uncertainty_avg = uncertainty_sum/total

    return loss_avg, mIoU_avg, uncertainty_avg, fpr, tpr, auc_value
    


'''
One epoch of inference, store all the preds, compute roc curve, compute pred maps 
'''
def seg_inference_roc(post_process_bool, model, criterion, dataloader, norm_size, num_batch, kid, bayes_bool, num_runs_bayes, device, infer_type, last_epoch=False):
    assert infer_type == 'test' or 'val', 'No inference type'
    loss_infer = 0
    y_prob_all = [] # (N,512,512)
    y_prob_var_all = [] # (N,2,512,512)
    y_gt_all = [] # (N,512,512)
    targets_origin_all = []
    filename_all = []
    with torch.no_grad():
        # if patch based: inputs(1,3,768,768) targets(1,768,768) targets_origins(1,origin size)
        for batch_idx, (inputs, targets, targets_origins, filenames) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if bayes_bool:
                # model uncertainty analysis
                loss_bayes_total = []
                probs_total = []
                for run in range(num_runs_bayes):
                    if norm_size != 512:
                        outputs = patch_infer_output(model, inputs, norm_size, device)
                    else:    
                        outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    loss_bayes_total.append(loss.item())

                    probs = F.softmax(outputs, dim=1)
                    probs_total.append(probs.cpu().detach().numpy())
                
                loss_infer += np.nanmean(loss_bayes_total)
                y_probs = np.nanmean(probs_total, axis=0) # (1,2,512,512)
                y_probs_var = np.var(probs_total, axis=0)
                y_prob_all.append(y_probs[:,1,:,:])
                y_prob_var_all.append(y_probs_var[0])
                
            else:
                if norm_size != 512:
                    outputs = patch_infer_output(model, inputs, norm_size, device)
                else:    
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_infer += loss.item()
                y_probs = F.softmax(outputs, dim=1)
                y_prob_all.append(y_probs[:,1,:,:].cpu().detach().numpy())

            y_gt_all.append(targets[0].cpu().detach().numpy()) 
            targets_origin_all.append(targets_origins[0].detach().numpy())
            filename_all.append(filenames[0])

    y_prob_all = np.array(y_prob_all)
    y_gt_all = np.array(y_gt_all)
    fpr, tpr, threshs = roc_curve(y_gt_all.reshape(-1), y_prob_all.reshape(-1))
    auc_value = auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1-fpr))
    id = np.argmax(gmeans)
    thresh_opt = threshs[id]
    y_pred_all = np.where(y_prob_all<thresh_opt, 0, 1) # (N,512,512)

    mIoU_sum = 0
    uncertainty_sum = 0
    for i in range(num_batch):
        # restore to original mask image size then compute miou
        y_pred = y_pred_all[i][0]
        target_origin = targets_origin_all[i]
        target = y_gt_all[i]
        filename = filename_all[i]

        var_map = None
        if bayes_bool:
            y_prob_var = y_prob_var_all[i]
            # generate the uncertainty map (probs_var, preds)
            var_map = np.take_along_axis(y_prob_var, y_pred[:,None],axis=0)[:,0]
            uncertainty_sum += np.sum(var_map)

        # post process
        if post_process_bool and infer_type == 'test':
            y_pred = post_process(y_pred, norm_size)

        pred_restore = image_restore_seg(y_pred, target_origin.shape)
        mscore = mIoU_score(pred_restore, target_origin)

        if infer_type == 'test' or last_epoch:
            img_visual_name = 'fold%d_%s_pred%d_miou%.3f_%s.png'%(kid+1,infer_type,i+1,mscore,filename)
            seg_visual(pred_restore, var_map, mscore, filename, img_visual_name, 'seg')
            seg_visual_overlap(norm_size, target, y_pred, var_map, filename, img_visual_name, 'seg')
    
        mIoU_sum += mscore

    
    loss_avg = loss_infer/num_batch
    mIoU_avg = mIoU_sum/num_batch
    uncertainty_avg = uncertainty_sum/num_batch

    if infer_type == 'val':
        prefix = 'Valid Epoch'
    else:
        prefix = 'Test Model'
    if bayes_bool:
        print('%s: loss=%.4f mIoU=%.3f AUC=%.3f threshold=%.3f uncertainty=%.3f' % (prefix, loss_avg, mIoU_avg, auc_value, thresh_opt, uncertainty_avg))
    else:
        print('%s: loss=%.4f mIoU=%.3f AUC=%.3f threshold=%.3f' % (prefix, loss_avg, mIoU_avg, auc_value, thresh_opt))

    return loss_avg, mIoU_avg, uncertainty_avg, fpr, tpr, auc_value, thresh_opt
    

def seg_data_path_load():
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
    img_paths, mask_paths = shuffle(img_paths, mask_paths, random_state=39)
    assert len(img_paths) == len(mask_paths), 'image and mask number don\'t match'
    for i in range(len(img_paths)):
        img_name = os.path.split(img_paths[i])[-1].split('.')[0]
        mask_name = os.path.split(mask_paths[i])[-1].split('.')[0]
        assert img_name == mask_name, "image and mask don't match"
    
    return img_paths, mask_paths 

if __name__ == '__main__':

    val_ratio = 0.2
    N_FOLD = 5
    BATCHSIZE_TE = 1
    LR = 0.001
    # wce, bce, dice, dicebce
    loss_type = 'dicebce'
    # unet, segnet
    model_type ='segnet'

    BATCHSIZE_TR = 1
    EPOCH = 1

    thresh_auto = False
    thresh_value = 0.002
    
    NORM_SIZE = 512
    aug_train_bool = True
    # ['crop_stride', 'rotate', 'flip']
    aug_list = ['flip']
    clahe_bool = True
    post_process_bool = True

    bayes_bool = True # turn on dropout or not
    dropout = 0.2      # dropout rate
    dropout_state = (False, False, True)  # turn on which layers
    num_runs_bayes = 20
    

    print('Loading data...')
    img_paths, mask_paths = seg_data_path_load()

    # miou of the last epoch
    val_miou_per_fold = np.zeros(N_FOLD)
    train_miou_per_fold = np.zeros(N_FOLD)
    # miou of the best epoch
    val_miou_best_per_fold = np.zeros(N_FOLD)
    best_model_paths = []
    
    uncertainty_last_per_fold = np.zeros(N_FOLD)
    uncertainty_best_per_fold = np.zeros(N_FOLD)

    tpr_per_fold = {} # key:kid, value tpr
    fpr_per_fold = {}
    auc_per_fold = np.zeros(N_FOLD)
    thresh_per_fold = np.zeros(N_FOLD)

    ### Split dataset into training and test set according to cross validation
    kf = KFold(n_splits=N_FOLD)
    for k_i, (trv_index, test_index) in enumerate(kf.split(img_paths)):
        print(f'\n{k_i+1} fold model')
        trv_img_paths, trv_mask_paths = data_extract(trv_index, img_paths, mask_paths)
        test_img_paths, test_mask_paths = data_extract(test_index, img_paths, mask_paths)

        ## Train/Val split
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(trv_img_paths, trv_mask_paths, test_size=val_ratio, shuffle=False)
        
        print(f'Train size: {len(train_img_paths)}')
        print(f'Test size: {len(test_img_paths)}')
        print(f'Val size: {len(val_img_paths)}')
        
        ## Create hdf5 file for each split
        print('Generating HDF5 file...')
        t1 = time.time()
        create_h5_train_seg(train_img_paths, train_mask_paths, 'train', NORM_SIZE, clahe_bool)
        create_h5_test_seg(test_img_paths, test_mask_paths, 'test', NORM_SIZE, clahe_bool, fid=k_i+1)
        create_h5_test_seg(val_img_paths, val_mask_paths, 'val', NORM_SIZE, clahe_bool)
        t2 = time.time()
        print('HDF5 file generation time: %.2f (min)' % ((t2-t1)/60))

        trainset = AFDataset('train', augmentation=aug_train_bool, aug_list=aug_list)
        valset = AFDataset('val', augmentation=False)

        trainloader = DataLoader(trainset, batch_size=BATCHSIZE_TR, shuffle=True, num_workers=0)
        validloader = DataLoader(valset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # print("Device: ", torch.cuda.get_device_name(device))
            torch.cuda.empty_cache()
        
        if model_type == 'unet':
            segmodel = UNet().to(device)
        elif model_type == 'segnet':
            segmodel = SegNet(bayes_bool, dropout, dropout_state).to(device)
        
        if loss_type == 'wce':
            weights = torch.tensor([1.0, 18.0]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        elif loss_type == 'bce':
            criterion = nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            criterion = DiceLoss()
        elif loss_type == 'dicebce':
            criterion = DiceBCELoss()

        optimizer = optim.SGD(segmodel.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.Adam(segmodel.parameters(), lr=LR, weight_decay=5e-4)

        num_batch_train = len(trainloader)
        num_batch_val = len(validloader)
        # num_batch_test = len(testloader)

        print('\nTraining...')
        # save the train/valid loss/metric downsampled valid metric for every epoch 
        history = torch.zeros((EPOCH, 4))
        val_miou_best = 0.0
        model_best_path = None

        for epoch in range(EPOCH):
            if epoch == EPOCH-1:
                last_epoch = True
            else:
                last_epoch = False
            ################### Training ###################
            segmodel.train()
            # accumulate loss and mIoU over batches
            train_loss = 0
            mIoU_sum = 0
            total = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Train Epoch {epoch+1}")
                for batch_idx, (inputs, targets, _, _) in enumerate(tepoch):
                    batch_num = targets.size(0)
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()

                    outputs = segmodel(inputs)
                    loss = criterion(outputs, targets)
                
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    probs = F.softmax(outputs, dim=1)
                    preds = torch.where(probs[:,1,:,:]>thresh_value, 1, 0)
                    mIoU_sum += sum([mIoU_score(preds[i], targets[i]) for i in range(batch_num)]) 
                    total += batch_num

                    tepoch.set_postfix(loss=train_loss/(batch_idx+1), mIoU=100.*mIoU_sum/total)
            
            history[epoch][0] = train_loss / num_batch_train
            history[epoch][1] = mIoU_sum / total
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ################### Validation ################### 
            segmodel.eval()
            # test with dropout
            if bayes_bool:
                segmodel.apply(apply_dropout)

            ## Valid per epoch
            if thresh_auto:
                loss_val, mIoU_val, uncertainty_val, fpr, tpr, auc_value, thresh_opt = seg_inference_roc(post_process_bool, segmodel, criterion, validloader, NORM_SIZE, num_batch_val, k_i, bayes_bool, num_runs_bayes, device, 'val', last_epoch)
            else:
                loss_val, mIoU_val, uncertainty_val, fpr, tpr, auc_value = seg_inference(post_process_bool, thresh_value, segmodel, criterion, validloader, NORM_SIZE, num_batch_val, k_i, bayes_bool, num_runs_bayes, device, 'val', last_epoch)
            
            # uncertainty for every epoch
            if bayes_bool and last_epoch:
                    uncertainty_last_per_fold[k_i] = uncertainty_val

            history[epoch][2] = loss_val
            history[epoch][3] = mIoU_val

            ## Save the model(epoch) with highest val miou during training 
            ## Save fpr, tpr, auc_value, threshold of this epoch as the optimal result
            if mIoU_val > val_miou_best:
                if model_best_path is not None:
                    os.remove(model_best_path)
                modelpath = join(SEG_MODEL_PATH, 'fold%d_epoch%d_%.3f_lite.pt' % (k_i+1, epoch+1, mIoU_val))
                torch.save({'model_type': model_type,
                            'loss_type': loss_type,
                            'model_sd': segmodel.state_dict(),
                            'post_process_bool': post_process_bool,
                            'bayes_bool': bayes_bool,
                            'dropout_rate': dropout,
                            'dropout_state': dropout_state,
                            'num_runs_bayes': num_runs_bayes,
                            'norm_size': NORM_SIZE,
                            'thresh_auto': thresh_auto,
                            'thresh_value': thresh_value
                            }, modelpath)

                val_miou_best = mIoU_val
                model_best_path = modelpath
                
                tpr_per_fold[k_i] = tpr
                fpr_per_fold[k_i] = fpr
                auc_per_fold[k_i] = auc_value
                if thresh_auto:
                    thresh_per_fold[k_i] = thresh_opt

                if bayes_bool:
                    uncertainty_best_per_fold[k_i] = uncertainty_val

        train_miou_per_fold[k_i] = history[-1,1]
        # val_miou_ds_per_fold[k_i] = history[-1,-1]
        val_miou_per_fold[k_i] = history[-1,3]
        val_miou_best_per_fold[k_i] = val_miou_best
        best_model_paths.append(split(model_best_path)[-1])
        # if not save_best_model:
        #     # torch.save(history, join(SEG_MODEL_PATH, f'history{k_i+1}.pt'))
        #     torch.save(segmodel.state_dict(), join(SEG_MODEL_PATH, 'fold%d_%.3f_lite.pt' % (k_i+1, history[-1,-2])))


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
        # plt.plot(x, history[:,4], label='downsampled val mIoU') 
        plt.legend()
        plt.title('Training and validation mIoU for {} epochs'.format(EPOCH))
        plt.savefig(join(SEG_MODEL_PATH, f'train_val_miou{k_i+1}.png'))

        trainset.close()
        valset.close()
        # testset.close()
        # del train_img_paths, train_mask_paths, val_img_paths, val_mask_paths


    print('\nTrain finished.')
    stat_compute_log(train_miou_per_fold, 'train mIoU')
    stat_compute_log(val_miou_per_fold, 'validation mIoU')

    stat_compute_log(val_miou_best_per_fold, 'optimal validation mIoU')

    auc_avg_fold, auc_std_fold = stat_compute_log(auc_per_fold, 'AUC value')
    if thresh_auto:
        stat_compute_log(thresh_per_fold, 'optimal threshold')

    ### Plot roc curves for all folds including mean curve
    tpr_per_fold = np.array([tpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    fpr_per_fold = np.array([fpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    roc_plot_all(fpr_per_fold, tpr_per_fold, auc_avg_fold, auc_std_fold, 'val')

    if bayes_bool:
        stat_compute_log(uncertainty_last_per_fold, 'uncertainty')
        stat_compute_log(uncertainty_best_per_fold, 'optimal uncertainty')
    print('Best models: ', best_model_paths)
