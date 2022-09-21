from utils import *
from dataset import ASDataset
from Resnet_cla import ResNet18


def cla_inference_roc(model, criterion, dataloader, num_batch, device, infer_type):
    assert infer_type == 'test' or 'val', 'No inference type'
    loss_infer = 0
    total = 0

    # collect the probs for that epoch
    y_gt_all = []
    y_prob_all = []
    with torch.no_grad():
        # with tqdm(dataloader, unit="batch") as vepoch:
        # outputs (16,2) targets 16
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # vepoch.set_description(f"Valid Epoch {epoch+1}")

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_infer += loss.item()
            total += targets.size(0)

            probs = F.softmax(outputs, dim=1)[:,1] # 16
            y_prob_all.append(probs.cpu().detach().numpy())
            y_gt_all.append(targets.cpu().detach().numpy())

            # vepoch.set_postfix(loss=loss_infer/(batch_idx+1))
    
    y_gt_all = np.concatenate(y_gt_all).reshape(-1)
    # y_val_preds = np.concatenate(y_val_preds).reshape(-1)
    y_prob_all = np.concatenate(y_prob_all).reshape(-1)

    fpr, tpr, threshs = roc_curve(y_gt_all, y_prob_all)
    auc_value = auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1-fpr))
    id = np.argmax(gmeans)
    thresh_opt = threshs[id]
    y_pred_all = np.where(y_prob_all<thresh_opt, 0, 1) # (N,512,512)
    
    accuracy = accuracy_score(y_gt_all, y_pred_all)
    f1 = f1_score(y_gt_all, y_pred_all)
    loss_avg = loss_infer/num_batch

    if infer_type == 'val':
        prefix = 'Valid Epoch'
    else:
        prefix = 'Test Model'
    print('%s: loss=%.4f accuracy=%.3f, AUC=%.3f, F1 score=%.3f threshold=%.3f' % (prefix, loss_avg, accuracy, auc_value, f1, thresh_opt))

    return loss_avg, accuracy, f1, fpr, tpr, auc_value, thresh_opt

def cla_data_path_load():
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

    img_path_all, label_all = shuffle(img_path_all, label_all, random_state=39)
    assert len(img_path_all) == len(label_all), 'image and label size does not match'

    return img_path_all, label_all

if __name__ == '__main__':

    BATCHSIZE = 20
    LR = 0.1
    EPOCH = 80

    aug_train = False
    val_ratio = 0.2
    N_FOLD = 5

    '''
    Fetch the data from disk only when necessary
    '''
    print('Loading data...')

    img_path_all, label_all = cla_data_path_load()
    
    transform_train_aug = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomRotation((-5,5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_acc_ks = np.zeros(N_FOLD)
    val_acc_best_ks = np.zeros(N_FOLD)
    best_model_paths = []
    cm_logs = []
    tpr_per_fold = {} # key:kid, value tpr
    fpr_per_fold = {}
    auc_per_fold = np.zeros(N_FOLD)
    thresh_per_fold = np.zeros(N_FOLD)
    f1_per_fold = np.zeros(N_FOLD)

    kf = KFold(n_splits=N_FOLD)
    for k_i, (trv_index, test_index) in enumerate(kf.split(img_path_all)):
        print(f'\n{k_i+1} fold model')
        trv_img_paths, trv_labels = data_extract(trv_index, img_path_all, label_all)
        test_img_paths, test_labels = data_extract(test_index, img_path_all, label_all)

        ## Train/Val split
        train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(trv_img_paths, trv_labels, test_size=val_ratio, shuffle=False)
        
        print(f'Train size: {len(train_img_paths)}')
        print(f'Test size: {len(test_img_paths)}')
        print(f'Val size: {len(val_img_paths)}')


        # Create HDF5 file for each split
        print('Generating HDF5 file...')
        t1 = time.time()
        create_h5_cla(train_img_paths, train_labels, 'train')
        create_h5_cla(test_img_paths, test_labels, 'test', fid=k_i+1)
        create_h5_cla(val_img_paths, val_labels, 'val')
        t2 = time.time()
        print('HDF5 file generation time: %.2f (min)' % ((t2-t1)/60))

        
        if aug_train: 
            trainset = ASDataset('train', transform_train_aug)
        else:
            trainset = ASDataset('train', transform_train)
        valset = ASDataset('val', transform_test)
        # testset = ASDataset('test', transform_test)

        trainloader = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
        validloader = DataLoader(valset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
        # testloader = DataLoader(testset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)

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
        # num_batch_test = len(testloader)

        print('\nTraining...')
        # best_acc = 0
        # save the train/valid loss/metric for every epoch 
        history = torch.zeros((EPOCH, 4))
        val_acc_best = 0.0
        model_best_path = None

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
            
            loss_val, acc_val, f1score, fpr, tpr, auc_value, thresh_opt = cla_inference_roc(resnet18, criterion, validloader, num_batch_val, device, 'val')

            history[epoch][2] = loss_val
            history[epoch][3] = acc_val
            
            # Save the model(epoch) with the best acc during training
            if history[epoch][3] > val_acc_best:
                if model_best_path is not None:
                    os.remove(model_best_path)
                modelpath = join(CLA_MODEL_PATH, 'fold%d_epoch%d_%.3f_lite.pt' % (k_i+1, epoch+1, history[epoch][-1]))
                torch.save(resnet18.state_dict(), modelpath)
                val_acc_best = history[epoch][3]
                model_best_path = modelpath

                tpr_per_fold[k_i] = tpr
                fpr_per_fold[k_i] = fpr
                auc_per_fold[k_i] = auc_value
                thresh_per_fold[k_i] = thresh_opt
                f1_per_fold[k_i] = f1score
        
        val_acc_ks[k_i] = history[-1,-1]
        val_acc_best_ks[k_i] = val_acc_best
        best_model_paths.append(split(model_best_path)[-1])



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
        # testset.close()
        # del train_imgs, train_labels, val_imgs, val_labels


    print('\nTrain finished.')
    stat_compute_log(val_acc_ks, 'Validation accuracy')
    stat_compute_log(val_acc_best_ks, 'Validation best accuracy')

    auc_avg_fold, auc_std_fold = stat_compute_log(auc_per_fold, 'AUC value')
    stat_compute_log(thresh_per_fold, 'optimal threshold')
    stat_compute_log(f1_per_fold, 'f1 score')

    ### Plot roc curves for all folds including mean curve
    tpr_per_fold = np.array([tpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    fpr_per_fold = np.array([fpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    roc_plot_all(fpr_per_fold, tpr_per_fold, auc_avg_fold, auc_std_fold, 'val')

    print('Best models: ', best_model_paths)
    # for str in cm_logs:
    #     print(str)

