# Summer Project
### Classification of SLO and AF
`ClaNet.py`: VGG-like model. <br>
`Resnet_cla.py`: adapted ResNet model. <br>
`cla_train.py` `cla_test.py`: training and test. <br>

### Segmentation of FAF images
`SegNet.py`: SegNet model and enabling dropout to be bayesian SegNet. <br>
`UNet.py`: UNet model. <br>
`seg_train.py` `seg_test.py`: training and test. Some settings in `seg_train.py`: <br>


### Cross-stitch Network for FAF image segmentation (Multi-task learning)
`mtl_model.py`: adapted cross-stitch network accepting two different image sources as input.<br>



`dataset.py`: self-defined dataset for SLO-AF and FAF image sets. <br>
`utils.py`: helper functions for classification and segmentation.



