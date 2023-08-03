from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class FontClassifier(nn.Module):
    def __init__(self,use_fc = True):
        super().__init__()
        self.conv1  = nn.Conv2d(3,16,7)#3x128x128->16*122*122
        self.conv2  = nn.Conv2d(16,32,7)#16*122*122->32*116*116
        self.conv3  = nn.Conv2d(32,64,5)#32*116*116->64*112*112
        self.pool1 = nn.MaxPool2d(2)#64*112*112->64*56*56
        self.resblock2 = nn.Sequential(
            PreActResNetBlock(64,nn.ReLU),#64*56*56->64*56*56
            PreActResNetBlock(64,nn.ReLU)#64*56*56->64*56*56
        )
        self.resblock3 = nn.Sequential(
            PreActResNetBlock(64,nn.ReLU,True,128),#64*56*56->128*28*28
            PreActResNetBlock(128,nn.ReLU)#128*28*28->128*28*28
        )
        self.resblock4 = nn.Sequential(
            PreActResNetBlock(128,nn.ReLU,True,256),#128*28*28->256*14*14
            PreActResNetBlock(256,nn.ReLU)#256*14*14->256*14*14
        )
        self.resblock5 = nn.Sequential(
            PreActResNetBlock(256,nn.ReLU,True,512),#256*14*14->512*7*7
            PreActResNetBlock(512,nn.ReLU)#512*7*7->512*7*7
        )
        self.pool_final = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(512,499)
        self.use_fc = use_fc
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = torch.squeeze(torch.squeeze(self.pool_final(x),-1),-1)
        if self.use_fc:
            x = self.fc_final(x)
        return x
class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = (
            nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False))
            if subsample
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out
