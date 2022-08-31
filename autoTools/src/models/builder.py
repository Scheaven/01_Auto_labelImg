'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 11:04:44
LastEditors: Scheaven
LastEditTime: 2022-07-29 17:23:20
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
        t1 = time.time()
        feats = self.backbone(x)
        # print(feats)
        # exit(0)
        t2 = time.time()
        cls, loc = self.rpn_head(self.temp_feats, feats)
        # print("banck time:", (time.time()-t2)*1000, (t2-t1)*1000)
        # print(cls.size())
        # cls = cls.permute(1,2,3,0).contiguous().view(2,-1).permute(1, 0)
        # score = F.softmax(cls, dim=1)[:,1]
        # delta = loc.permute(1,2,3,0).contiguous().view(4,-1)
        # print(score)
        # print("---------------------------")
        # exit(0)
        return {
            'cls': cls,
            'loc': loc
        }
        
    def forward(self, template):
        xf = self.backbone(template)
        print(xf.size())
        return xf
