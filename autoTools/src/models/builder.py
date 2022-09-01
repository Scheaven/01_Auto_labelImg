'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 11:04:44
LastEditors: Scheaven
LastEditTime: 2022-09-01 11:03:16
Remark: Never cease to pursue!
'''
import torch.nn as nn
import torch.nn.functional as F
from autoTools.src.models.backbone import Backbone
from autoTools.src.models.head import MulHead
import time
class ModelBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.rpn_head = MulHead()
        self.temp_list = []

    def template(self, x):
        self.temp_feats = self.backbone(x)
        return self.temp_feats 

    def track(self, x):
        feats = self.backbone(x)
        cls, loc = self.rpn_head(self.temp_feats, feats)
        return {
            'cls': cls,
            'loc': loc
        }
        
    def forward(self, template):
        xf = self.backbone(template)
        return xf
