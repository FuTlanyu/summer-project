import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary as pms



class Cross_stitch(nn.Module):
    def __init__(self, channels, p_same, p_diff):
        super().__init__()

        # define trainable parameters
        self.channels = channels
        # p_diff = 1-p_same
        self.p_same, self.p_diff = p_same, p_diff

        weight_cs = np.array([[p_same, p_diff],[p_diff, p_same]])
        weights = np.repeat(weight_cs[np.newaxis, :, :], self.channels, axis=0)
        self.weights = nn.Parameter(torch.Tensor(weights))

    def forward(self, x_a, x_b, device):
        
        # input size (N, C, H, W)
        assert x_a.shape == x_b.shape, f"tensor shape a ({x_a.shape}) and b ({x_b.shape}) doesn't match"
        size = x_a.shape
        N,C,H,W = size
        x_a_cs = torch.empty(size=size, dtype=torch.float32).to(device)
        x_b_cs = torch.empty(size=size, dtype=torch.float32).to(device)
        
        # for every channel, the module use the same set of weights
        for c in range(C):
            x_a_chn = x_a[:,c,:,:]  # (N, H, W)
            x_b_chn = x_b[:,c,:,:] 
            weight = self.weights[c,:,:] # (2,2)

            x_a_chn_reshaped = torch.unsqueeze(torch.flatten(x_a_chn), dim=0)
            x_b_chn_reshaped = torch.unsqueeze(torch.flatten(x_b_chn), dim=0)
            x_combo_chn = torch.concat((x_a_chn_reshaped, x_b_chn_reshaped), dim=0)
            x_combo_chn_cs = torch.matmul(weight, x_combo_chn)

            # restore
            x_a_chn_cs = x_combo_chn_cs[0].reshape((N, H, W))
            x_b_chn_cs = x_combo_chn_cs[1].reshape((N, H, W))
            x_a_cs[:,c,:,:] = x_a_chn_cs
            x_b_cs[:,c,:,:] = x_b_chn_cs

        return x_a_cs, x_b_cs



class MTL(nn.Module):
    def block(self, conv_num, in_chn, out_chn):
        layers = []
        for i in range(conv_num):
            if i == 0:
                layers.append(nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_chn))
            layers.append(nn.ReLU())   
        
        return nn.Sequential(*layers)
    
    def de_block(self, conv_num, in_chn, out_chn):
        layers = []
        for i in range(conv_num):
            if i == conv_num-1:
                layers.append(nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(out_chn))
            else:
                layers.append(nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(in_chn))
            layers.append(nn.ReLU())   
        
        return nn.Sequential(*layers)

    def __init__(self, p_same, p_diff, device, bayes=False, dropout=0.5, classes=2):
        super(MTL, self).__init__()
        self.classes = classes
        self.bayes = bayes
        self.dropout = dropout
        self.p_same, self.p_diff = p_same, p_diff
        self.device = device

        if self.bayes:
            self.dropout = nn.Dropout(p=self.dropout)
        # Segmentation encoder
        self.maxEn_1 = nn.MaxPool2d(2, stride=2, return_indices=True) 
        self.conv_block1_1 = self.block(conv_num=2, in_chn=3, out_chn=64)
        self.conv_block2_1 = self.block(conv_num=2, in_chn=64, out_chn=128)
        self.conv_block3_1 = self.block(conv_num=3, in_chn=128, out_chn=256)
        self.conv_block4_1 = self.block(conv_num=3, in_chn=256, out_chn=512)
        self.conv_block5_1 = self.block(conv_num=3, in_chn=512, out_chn=512)

        # Cross stitch units
        self.cross_stitch1 = Cross_stitch(channels=64, p_same=self.p_same, p_diff=self.p_diff)
        self.cross_stitch2 = Cross_stitch(channels=128, p_same=self.p_same, p_diff=self.p_diff)
        self.cross_stitch3 = Cross_stitch(channels=256, p_same=self.p_same, p_diff=self.p_diff)
        self.cross_stitch4 = Cross_stitch(channels=512, p_same=self.p_same, p_diff=self.p_diff)
        self.cross_stitch5 = Cross_stitch(channels=512, p_same=self.p_same, p_diff=self.p_diff)

        # Classification encoder
        self.maxEn_2 = nn.MaxPool2d(2, stride=2, return_indices=False) 
        self.conv_block1_2 = self.block(conv_num=2, in_chn=3, out_chn=64)
        self.conv_block2_2 = self.block(conv_num=2, in_chn=64, out_chn=128)
        self.conv_block3_2 = self.block(conv_num=3, in_chn=128, out_chn=256)
        self.conv_block4_2 = self.block(conv_num=3, in_chn=256, out_chn=512)
        self.conv_block5_2 = self.block(conv_num=3, in_chn=512, out_chn=512)



        # Segmentation decoder
        self.maxDe = nn.MaxUnpool2d(2, stride=2)
        self.deconv_block1 = self.de_block(conv_num=3, in_chn=512, out_chn=512)
        self.deconv_block2 = self.de_block(conv_num=3, in_chn=512, out_chn=256)
        self.deconv_block3 = self.de_block(conv_num=3, in_chn=256, out_chn=128)
        self.deconv_block4 = self.de_block(conv_num=2, in_chn=128, out_chn=64)
        # self.deconv_block5 = self.de_block(conv_num=2, in_chn=64, out_chn=self.classes)
        self.deconv_block5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.classes, kernel_size=3, stride=1, padding=1),
        )

        # Classification fully connection
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x_seg, x_cla):
        """forward propogation of multi-task learning model

        Args:
            x_seg: input for segmentation
            x_cla: input for classification

        Returns:
            _description_
        """

        # Encoder
        x_seg = self.conv_block1_1(x_seg)  # conv/bn/relu*2
        size1 = x_seg.size()
        x_seg, ind1 = self.maxEn_1(x_seg)

        x_cla = self.conv_block1_2(x_cla)
        x_cla = self.maxEn_2(x_cla)
        # cross stitch 1
        x_seg, x_cla = self.cross_stitch1(x_seg, x_cla, self.device)


        x_seg = self.conv_block2_1(x_seg)
        size2 = x_seg.size()
        x_seg, ind2 = self.maxEn_1(x_seg)

        x_cla = self.conv_block2_1(x_cla)
        x_cla = self.maxEn_2(x_cla)
        # cross stitch 2
        x_seg, x_cla = self.cross_stitch2(x_seg, x_cla, self.device)


        x_seg = self.conv_block3_1(x_seg)
        size3 = x_seg.size()
        x_seg, ind3 = self.maxEn_1(x_seg)

        x_cla = self.conv_block3_1(x_cla)
        x_cla = self.maxEn_2(x_cla)
        # cross stitch 3
        x_seg, x_cla = self.cross_stitch3(x_seg, x_cla, self.device)
        # dropout 3
        if self.bayes:
            x_seg = self.dropout(x_seg)
            x_cla = self.dropout(x_cla)


        x_seg = self.conv_block4_1(x_seg)
        size4 = x_seg.size()
        x_seg, ind4 = self.maxEn_1(x_seg)
        
        x_cla = self.conv_block4_1(x_cla)
        x_cla = self.maxEn_2(x_cla)
        # cross stitch 4
        x_seg, x_cla = self.cross_stitch4(x_seg, x_cla, self.device)

        # dropout 4
        if self.bayes:
            x_seg = self.dropout(x_seg)
            x_cla = self.dropout(x_cla)


        x_seg = self.conv_block5_1(x_seg)
        size5 = x_seg.size()
        x_seg, ind5 = self.maxEn_1(x_seg)

        x_cla = self.conv_block5_1(x_cla)
        x_cla = self.maxEn_2(x_cla)
        # cross stitch 5
        x_seg, x_cla = self.cross_stitch5(x_seg, x_cla, self.device)
        
        # dropout 5
        if self.bayes:
            x_seg = self.dropout(x_seg)
            x_cla = self.dropout(x_cla)




        #### Segmentation decoder ####
        x_seg = self.maxDe(x_seg, ind5, output_size=size5)
        x_seg = self.deconv_block1(x_seg)
        # dropout 6
        if self.bayes:
            x_seg = self.dropout(x_seg)

        x_seg = self.maxDe(x_seg, ind4, output_size=size4)
        x_seg = self.deconv_block2(x_seg)
        # dropout 7
        if self.bayes:
            x_seg = self.dropout(x_seg)

        x_seg = self.maxDe(x_seg, ind3, output_size=size3)
        x_seg = self.deconv_block3(x_seg)
        # dropout 8
        if self.bayes:
            x_seg = self.dropout(x_seg)

        x_seg = self.maxDe(x_seg, ind2, output_size=size2)
        x_seg = self.deconv_block4(x_seg)

        x_seg = self.maxDe(x_seg, ind1, output_size=size1)
        x_seg = self.deconv_block5(x_seg)

        
        #### Classification fully connection ####
        x_cla = self.avgpool(x_cla)
        x_cla = x_cla.reshape(x_cla.shape[0], -1)
        x_cla = self.fc(x_cla)
        
        return x_seg, x_cla

# # print(pms.summary(MTL(), torch.zeros((10, 3, 256, 256)), show_input=True, show_hierarchical=True))
