# Summer Project
### Classification of SLO and AF
`ClaNet.py`: VGG-like model. <br>
`Resnet_cla.py`: adapted ResNet model. <br>
`cla_train.py` `cla_test.py`: training and test. <br>

### Segmentation of FAF images
`SegNet.py`: SegNet model and enabling dropout to be bayesian SegNet. <br>
`UNet.py`: UNet model. <br>
`seg_train.py` `seg_test.py`: training and test. Some settings in `seg_train.py`: <br>
- `val_ratio`: the proportion of the validation set in the development set.
- `loss_type`: loss functions. Optional loss functions are `wce` `bce` `dice` `dicebce`.
- `model_type`: segmentation networks. Optional models are `segnet` `unet`.
- `thresh_auto`: bool, threshold moving or not.
- `thresh_value`: float between 0 and 1, chosen threshold value for mapping probabilities to class labels.
- `NORM_SIZE`: int, uniform image size for model training.
- `aug_train_bool`: bool, training data augmentation or not.
- `aug_list`: list, a list of augmentation methods including `flip` `rotate` `crop_stride`. `crop_stride` adopted a patch-based image augmentation.
- `clahe_bool`: bool, CLAHE image enhancement or not.
- `post_process_bool`: bool, post-processing for predicted results or not.
- `bayes_bool`: bool, enabling dropout layers in the SegNet model to implement uncertainty analysis or not.
- `dropout`: float between 0 and 1, dropout rate, a larger value representing deactivating more pixels in feature maps.
- `dropout_state`: tuple of bools, determining which dropout pairs to be activated. For example, `(False, False, True)` means deactivating the first and second dropout pairs and activating the third pair.
- `num_runs_bayes`: int, number of runs during inference time when enabling dropout to generate uncertainty maps.

An optimal model with the Dice-BCE loss, Bayesian SegNet with a dropout rate of 0.2 on the third pair, image size of 512, using flipping and CLAHE with clip limit rate of 1.0, a fixed threshold of 0.002 with post-processing reached an test mIoU of 0.8709 ± 0.0261. The comparision of different models is shown in the table below.
![model_results](https://user-images.githubusercontent.com/36615950/191891610-937e7d5f-6089-4882-8902-1ff9dd066bd7.png)

Some predicted results under the optimal settings are illustrated below. The green contour is the ground truth and the red contour is the predicted mask. The heatmap represents the uncertainty of the prediction. 
![opt_vis](https://user-images.githubusercontent.com/36615950/191891915-3ede4dc2-e4c3-4339-8820-792e24732ea0.png)


### Cross-stitch Network for FAF image segmentation (Multi-task learning)
`mtl_model.py`: adapted cross-stitch network accepting two different image sources as input.<br>
`mtl_train.py` `mtl_test.py`: training and test. Some settings in `mtl_train.py`: <br>
- `p_same` `p_diff`: float between 0 and 1, initial cross-stitch unit weights.
- `compute_wcs`: bool, recording cross-stitch unit weights during training or not.
- `cs_diff_lr`: bool, adopting a different learning rate for cross-stitch unit weights or not.
- `CS_LR_RATIO`: learning rate for cross-stitch unit weights.
- `pretrain_load`: bool, loading pre-trained model weights for segmentation and classification sub-networks or not.
- `PRE_SEG_EPOCH` `PRE_CLA_EPOCH`: pre-training epochs for segmentation and classification respectively.



`dataset.py`: self-defined dataset for SLO-AF and FAF image sets. <br>
`utils.py`: helper functions for classification and segmentation.



