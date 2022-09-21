from utils import *
from dataset import ASDataset
from Resnet_cla import ResNet18
from cla_train import cla_inference_roc

if __name__ == '__main__':

    BATCHSIZE = 20
    N_FOLD = 5
    # dicebceloss

    print('\nTesting...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trained_model_paths = []
    for f in os.listdir(CLA_MODEL_PATH):
        if os.path.isfile(join(CLA_MODEL_PATH, f)) and 'lite' in f:
            trained_model_paths.append(join(CLA_MODEL_PATH, f))
    assert len(trained_model_paths) == N_FOLD, 'Trained model numbers and fold don\'t match' 
    trained_model_paths = sorted(trained_model_paths)

    ## Load N different models for testing
    loss_per_fold = []
    acc_per_fold = []
    tpr_per_fold = {} # key:kid, value tpr
    fpr_per_fold = {}
    auc_per_fold = np.zeros(N_FOLD)
    thresh_per_fold = np.zeros(N_FOLD)
    f1_per_fold = np.zeros(N_FOLD)
    # cm_logs = []
    for ki in range(N_FOLD):
        testset = ASDataset('test', transform_test, fid=ki+1)
        testloader = DataLoader(testset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
        num_batches = len(testloader)

        model_sd = torch.load(trained_model_paths[ki])
        resnet18 = ResNet18().to(device)
        resnet18.load_state_dict(model_sd)
        criterion = torch.nn.CrossEntropyLoss()

        resnet18.eval()

        loss_avg, accuracy, f1score, fpr, tpr, auc_value, thresh_opt = cla_inference_roc(resnet18, criterion, testloader, num_batches, device, 'test')

        loss_per_fold.append(loss_avg)
        acc_per_fold.append(accuracy)
        tpr_per_fold[ki] = tpr
        fpr_per_fold[ki] = fpr
        auc_per_fold[ki] = auc_value
        thresh_per_fold[ki] = thresh_opt
        f1_per_fold[ki] = f1score
        
        # tn, fp, fn, tp = confusion_matrix(y_test_trues, y_test_preds).ravel()
        # cm_log = f'Total test samples: {y_test_trues.shape[0]} TP: {tp} TN: {tn} FP: {fp} FN: {fn}'
        # cm_logs.append(cm_log)
        # fpr, tpr, _ = roc_curve(y_test_trues, y_test_scores)
        # roc_auc = auc(fpr, tpr)

        # roc_plot(fpr, tpr, roc_auc, ki)
        testset.close()
    
    print('\nTest finished.')    
    stat_compute_log(loss_per_fold, 'test loss')
    stat_compute_log(acc_per_fold, 'test acc')

    auc_avg_fold, auc_std_fold = stat_compute_log(auc_per_fold, 'AUC value')
    stat_compute_log(thresh_per_fold, 'optimal threshold')
    stat_compute_log(f1_per_fold, 'f1 score')

    ### Plot roc curves for all folds including mean curve
    tpr_per_fold = np.array([tpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    fpr_per_fold = np.array([fpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    roc_plot_all(fpr_per_fold, tpr_per_fold, auc_avg_fold, auc_std_fold, 'test')

    # for str in cm_logs:
    #     print(str)