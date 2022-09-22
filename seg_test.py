from utils import *
from SegNet import SegNet
from UNet import UNet
from dataset import AFDataset
from seg_train_train_test_cv import seg_inference, seg_inference_roc


if __name__ == '__main__':

    BATCHSIZE_TE = 1
    N_FOLD = 5

    print('\nTesting...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trained_model_paths = []
    for f in os.listdir(SEG_MODEL_PATH):
        if os.path.isfile(join(SEG_MODEL_PATH, f)) and 'lite' in f:
            trained_model_paths.append(join(SEG_MODEL_PATH, f))
    assert len(trained_model_paths) == N_FOLD, 'Trained model numbers and fold don\'t match' 
    trained_model_paths = sorted(trained_model_paths)

    ## Load N different models for testing
    loss_per_fold = []
    miou_per_fold = []
    uncertainty_per_fold = []
    tpr_per_fold = {} # key:kid, value tpr
    fpr_per_fold = {}
    auc_per_fold = []
    thresh_per_fold = []

    for ki in range(N_FOLD):
        testset = AFDataset('test', augmentation=False, fid=ki+1)
        testloader = DataLoader(testset, batch_size=BATCHSIZE_TE, shuffle=False, num_workers=0)
        num_batches = len(testloader)

        config = torch.load(trained_model_paths[ki])
        model_type = config['model_type']
        loss_type = config['loss_type']
        model_sd = config['model_sd']
        post_process_bool = config['post_process_bool']
        bayes_bool = config['bayes_bool']
        dropout = config['dropout_rate']
        dropout_state = config['dropout_state']
        num_runs_bayes = config['num_runs_bayes']
        norm_size = config['norm_size']
        thresh_auto = config['thresh_auto']
        thresh_value = config['thresh_value']

        if model_type == 'unet':
            segmodel = UNet().to(device)
        elif model_type == 'segnet':
            segmodel = SegNet(bayes_bool, dropout, dropout_state).to(device)

        segmodel.load_state_dict(model_sd)

        if loss_type == 'wce':
            weights = torch.tensor([1.0, 18.0]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        elif loss_type == 'bce':
            criterion = nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            criterion = DiceLoss()
        elif loss_type == 'dicebce':
            criterion = DiceBCELoss()

        segmodel.eval()
        # test with dropout
        if bayes_bool:
            segmodel.apply(apply_dropout)
        if thresh_auto:
            loss, miou, uncertainty, fpr, tpr, auc_value, thresh_opt = seg_inference_roc(post_process_bool, segmodel, criterion, testloader, norm_size, num_batches, ki, bayes_bool, num_runs_bayes, device, 'test')
        else:
            loss, miou, uncertainty, fpr, tpr, auc_value = seg_inference(post_process_bool, thresh_value, segmodel, criterion, testloader, norm_size, num_batches, ki, bayes_bool, num_runs_bayes, device, 'test')
        loss_per_fold.append(loss)
        miou_per_fold.append(miou)
        uncertainty_per_fold.append(uncertainty)
        fpr_per_fold[ki] = fpr
        tpr_per_fold[ki] = tpr
        auc_per_fold.append(auc_value)
        if thresh_auto:
            thresh_per_fold.append(thresh_opt)
        
        testset.close()

    print('\nTest finished.')    
    stat_compute_log(loss_per_fold, 'test loss')
    stat_compute_log(miou_per_fold, 'test mIoU')
    
    auc_avg_fold, auc_std_fold = stat_compute_log(auc_per_fold, 'test AUC value')
    if thresh_auto:
        stat_compute_log(thresh_per_fold, 'test optimal threshold')

    ### Plot roc curves for all folds including mean curve
    tpr_per_fold = np.array([tpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    fpr_per_fold = np.array([fpr_per_fold[i] for i in range(N_FOLD)], dtype=object)
    roc_plot_all(fpr_per_fold, tpr_per_fold, auc_avg_fold, auc_std_fold, 'test')
    
    if bayes_bool:
        stat_compute_log(uncertainty_per_fold, 'test uncertainty')
