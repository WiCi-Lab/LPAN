# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 19:51:09 2022

@author: WiCi
"""


import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.nn.utils import weight_norm


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=True),
            # nn.LeakyReLU(negative_slope=0.3),
            # nn.Linear(channel // reduction, channel, bias=True),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, x):
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class _Res_Blocka(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Res_Blocka, self).__init__()
        
        self.res_conv = weight_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.res_conb = weight_norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.ca = SELayer(out_ch)

    def forward(self, x,al=1):

        y = self.relu(self.res_conv(x))
        y = self.relu(self.res_conb(y))
        y = self.ca(y)
        y *= al
        y = torch.add(y, x)
        return y
    
class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        numf=96
        self.conv4 = weight_norm(nn.Conv2d(numf, numf, (3, 3), (1, 1), (1, 1)))

        self.convt_F = nn.Upsample(size=None, scale_factor=(1,2), mode='nearest', align_corners=None)

        self.LReLus = nn.LeakyReLU(negative_slope=0.2)
        
        m_body = [
            _Res_Blocka(numf,numf) for _ in range(4)
        ]
        
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
            
        out = self.body(x)
        out =  self.LReLus(self.conv4(self.convt_F(out)))

        return out

class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv_R = weight_norm(nn.Conv2d(96, 2, (3, 3), (1, 1), padding=1))
        self.convt_I = nn.Upsample(size=None, scale_factor=(1,2), mode='nearest', align_corners=None)
        self.conv_1 = nn.Conv2d(2, 2, (3, 3), (1, 1), padding=1)
        
    def forward(self, LR, convt_F):
        convt_I = self.conv_1(self.convt_I(LR))
        
        conv_R = self.conv_R(convt_F)
        
        HR = convt_I+conv_R
        
        return HR
        
        
class LPAN(nn.Module):
    def __init__(self):
        super(LPAN, self).__init__()
        numf=96
        self.conv0 = weight_norm(nn.Conv2d(2, numf, (3, 3), (1, 1), padding=1))
        self.FeatureExtraction1 = FeatureExtraction()
        self.FeatureExtraction2 = FeatureExtraction()
        self.FeatureExtraction3 = FeatureExtraction()
        self.ImageReconstruction1 = ImageReconstruction()
        self.ImageReconstruction2 = ImageReconstruction()
        self.ImageReconstruction3 = ImageReconstruction()


    def forward(self, LR):
        
        LR1 = self.conv0(LR)
        
        convt_F1 = self.FeatureExtraction1(LR1)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)
        
        convt_F2 = self.FeatureExtraction2(convt_F1)
        HR_4 = self.ImageReconstruction2(HR_2, convt_F2)
        
        convt_F3 = self.FeatureExtraction3(convt_F2)
        HR_8 = self.ImageReconstruction3(HR_4, convt_F3)
        
        return HR_2, HR_4, HR_8

        

    