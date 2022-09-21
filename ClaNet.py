import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary as pms


class ClaNet(nn.Module):
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

    def __init__(self, classes=2):
        super(ClaNet, self).__init__()
        self.classes = classes

        # Encoder
        self.maxEn = nn.MaxPool2d(2, stride=2, return_indices=False) 
        self.conv_block1 = self.block(conv_num=2, in_chn=3, out_chn=64)
        self.conv_block2 = self.block(conv_num=2, in_chn=64, out_chn=128)
        self.conv_block3 = self.block(conv_num=3, in_chn=128, out_chn=256)
        self.conv_block4 = self.block(conv_num=3, in_chn=256, out_chn=512)
        self.conv_block5 = self.block(conv_num=3, in_chn=512, out_chn=512)

        # Classification fully connection
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):

        # Encoder
        x = self.conv_block1(x)
        x = self.maxEn(x)
        
        x = self.conv_block2(x)
        x = self.maxEn(x)
        
        x = self.conv_block3(x)
        x = self.maxEn(x)
        # # dropout 3
        # if self.bayes:
        #     x = self.dropout(x)
        
        x = self.conv_block4(x)
        x = self.maxEn(x)
        # # dropout 4
        # if self.bayes:
        #     x = self.dropout(x)

        x = self.conv_block5(x)
        x = self.maxEn(x)
        # dropout 5
        # if self.bayes:
        #     x = self.dropout(x)

        #### Classification fully connection ####
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x


