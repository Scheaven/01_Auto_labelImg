'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 11:14:23
LastEditors: Scheaven
LastEditTime: 2022-07-28 17:12:38
Remark: Never cease to pursue!
'''
from typing import ForwardRef
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
        )
        self.feature_size = 256

    def forward(self, x):
        x = self.features(x)
        return x