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



### Cross-stitch Network for FAF image segmentation (Multi-task learning)
`mtl_model.py`: adapted cross-stitch network accepting two different image sources as input.<br>



`dataset.py`: self-defined dataset for SLO-AF and FAF image sets. <br>
`utils.py`: helper functions for classification and segmentation.



