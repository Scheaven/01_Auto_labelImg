'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 16:19:05
LastEditors: Scheaven
LastEditTime: 2022-03-08 18:03:39
Remark: Never cease to pursue!
'''

import math
from typing import Sized
import numpy as np

class Anchors:
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride*self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws*s
                h = hs*s
                
                self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1
    
