'''
Adapted from https://github.com/vinceecws/SegNet_PyTorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary as pms


class SegNet(nn.Module):
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

    def __init__(self, bayes=False, dropout=0.5, classes=2):
        super(SegNet, self).__init__()
        self.classes = classes
        self.bayes = bayes
        self.dropout = dropout

        if self.bayes:
            self.dropout = nn.Dropout(p=self.dropout)
        # Encoder
        self.maxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 
        self.conv_block1 = self.block(conv_num=2, in_chn=3, out_chn=64)
        self.conv_block2 = self.block(conv_num=2, in_chn=64, out_chn=128)
        self.conv_block3 = self.block(conv_num=3, in_chn=128, out_chn=256)
        self.conv_block4 = self.block(conv_num=3, in_chn=256, out_chn=512)
        self.conv_block5 = self.block(conv_num=3, in_chn=512, out_chn=512)

        # Decoder
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

    def forward(self, x):

        # Encoder
        x = self.conv_block1(x)
        size1 = x.size()
        x, ind1 = self.maxEn(x)
        
        x = self.conv_block2(x)
        size2 = x.size()
        x, ind2 = self.maxEn(x)
        
        x = self.conv_block3(x)
        size3 = x.size()
        x, ind3 = self.maxEn(x)
        # dropout 3
        if self.bayes:
            x = self.dropout(x)
        
        x = self.conv_block4(x)
        size4 = x.size()
        x, ind4 = self.maxEn(x)
        # dropout 4
        if self.bayes:
            x = self.dropout(x)

        x = self.conv_block5(x)
        size5 = x.size()
        x, ind5 = self.maxEn(x)
        # dropout 5
        if self.bayes:
            x = self.dropout(x)

        # Decoder
        x = self.maxDe(x, ind5, output_size=size5)
        x = self.deconv_block1(x)
        # dropout 6
        if self.bayes:
            x = self.dropout(x)

        x = self.maxDe(x, ind4, output_size=size4)
        x = self.deconv_block2(x)
        # dropout 7
        if self.bayes:
            x = self.dropout(x)

        x = self.maxDe(x, ind3, output_size=size3)
        x = self.deconv_block3(x)
        # dropout 8
        if self.bayes:
            x = self.dropout(x)

        x = self.maxDe(x, ind2, output_size=size2)
        x = self.deconv_block4(x)

        x = self.maxDe(x, ind1, output_size=size1)
        x = self.deconv_block5(x)

        # x = F.softmax(x, dim=1)

        return x

print(pms.summary(SegNet(classes=2), torch.zeros((1, 3, 256, 256)), show_input=True, show_hierarchical=False))
# m = SegNet(bayes=True)
# param = list(m.parameters())
# print()
# m(torch.randn(1,3,256,256)) 
