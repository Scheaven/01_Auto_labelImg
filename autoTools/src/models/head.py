'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 12:06:18
LastEditors: Scheaven
LastEditTime: 2022-09-01 11:03:33
Remark: Never cease to pursue!
'''
# from typing import ForwardRef
import torch, time
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super().__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )
        self.deconv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
                            
            nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            )

        self.deconv_search = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                                
                nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

    def feature_corr(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        t_kernel = kernel.contiguous().view(batch*channel, 1, kernel.size(2), kernel.size(3))
        n_xx = x.contiguous().view(1, -1, x.size(2), x.size(3))
        corr = F.conv2d(n_xx, t_kernel, groups=batch*channel)
        corr = corr.view(batch, channel, corr.size()[2], corr.size()[3])
        return corr

    def forward(self, kernel, search):
        search = self.conv_search(search)
        kernel = self.conv_kernel(kernel)
        feature = self.feature_corr(search, kernel)
        out = self.head(feature)
        return out

class MulHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = Head(256, 256, 2*5)
        self.loc = Head(256, 256, 4*5)

    def forward(self, z, x):
        cls = self.cls(z, x)
        loc = self.loc(z, x)
        return cls, loc