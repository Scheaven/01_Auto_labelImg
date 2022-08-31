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
        # print(":::::::::::", x.size())
        return x

        # # 深度卷积，通道数不变，用于缩小特征图大小,
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(3, 96, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.MaxPool2d(kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.MaxPool2d(kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(256),
        # )
        # self.feature_size = 256

    # def forward(self, x):
    #     # x = self.features(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     if x.size(3) < 20: # Adjust consistency 后期修改标签删除
    #         l = (x.size(3) - 9) // 2
    #         r = l + 9
    #         x = x[:, :, l:r, l:r]
    #     # print("x:::::::::",x.size())
    #     return x

    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3, bias=False),
    #         nn.BatchNorm2d(3),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(3, 96, kernel_size=1, stride=1, padding=0, bias=False),
    #         nn.BatchNorm2d(96),
    #         nn.MaxPool2d(kernel_size=3, stride=1),
    #         nn.ReLU(inplace=True),
    #     )

    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=False),
    #         nn.BatchNorm2d(96),
    #         nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, bias=False),
    #         nn.BatchNorm2d(256),
    #     )

    #     self.feature_size = 256

    # def forward(self, x):
    #     # x = self.features(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     if x.size(3) < 20: # Adjust consistency 后期修改标签删除
    #         l = (x.size(3) - 7) // 2
    #         r = l + 7
    #         x = x[:, :, l:r, l:r]
    #     return x

    # def __init__(self, width_mult=1):
    #     configs = [3, 96, 256, 384, 384, 256]
    #     super(Backbone, self).__init__()
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
    #         nn.BatchNorm2d(configs[1]),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.ReLU(inplace=True),
    #         )
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(configs[1], configs[2], kernel_size=5),
    #         nn.BatchNorm2d(configs[2]),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.ReLU(inplace=True),
    #         )
    #     self.layer3 = nn.Sequential(
    #         nn.Conv2d(configs[2], configs[3], kernel_size=3),
    #         nn.BatchNorm2d(configs[3]),
    #         nn.ReLU(inplace=True),
    #         )
    #     self.layer4 = nn.Sequential(
    #         nn.Conv2d(configs[3], configs[4], kernel_size=3),
    #         nn.BatchNorm2d(configs[4]),
    #         nn.ReLU(inplace=True),
    #         )

    #     self.layer5 = nn.Sequential(
    #         nn.Conv2d(configs[4], configs[5], kernel_size=3),
    #         nn.BatchNorm2d(configs[5]),
    #         )
    #     self.feature_size = configs[5]

    # def forward(self, x):
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.layer5(x)
    #     # print(":::::::::::", x.size())
    #     # if x.size(3) < 20: # Adjust consistency 后期修改标签删除
    #     #     l = (x.size(3) - 7) // 2
    #     #     r = l + 7
    #     #     x = x[:, :, l:r, l:r]
    #     return x