'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 12:06:18
LastEditors: Scheaven
LastEditTime: 2022-07-28 17:15:08
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
        kernel_list = []
        # t1 = time.time()
        search = self.conv_search(search)
        # search = self.deconv_search(search)
        # t2 = time.time()
        kernel = self.conv_kernel(kernel)
        # kernel = self.deconv_kernel(kernel)
        # t3 = time.time()
        feature = self.feature_corr(search, kernel)
        # t4 = time.time()
        out = self.head(feature)
        # print("?? time::", (time.time() - t4)*1000, (t4 -t3)*1000, (t3 -t2)*1000, (t2 -t1)*1000)
        return out

class MulHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = Head(256, 256, 2*5)
        self.loc = Head(256, 256, 4*5)

    def forward(self, z, x):
        # t1 = time.time()
        cls = self.cls(z, x)
        # t2 = time.time()
        loc = self.loc(z, x)
        # print("banckdddddd time::", (time.time() - t2)*1000, (t2 -t1)*1000)
        return cls, loc