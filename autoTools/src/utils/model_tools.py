'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 18:29:06
LastEditors: Scheaven
LastEditTime: 2022-09-01 11:05:42
Remark: Never cease to pursue!
'''
from time import time
import numpy as np
from numpy.core.fromnumeric import shape
import torch.nn.functional as F
import torch
import time

def convert_score(score):
    score = score.permute(1, 2, 3, 0)
    score = score.contiguous()
    score = score.view(2, -1)
    score = score.permute(1, 0)
    score = F.softmax(score, dim=1)
    score = score.data[:, 1]
    score = score.cpu()
    score = score.numpy()
    return score

def convert_confK(confidences):
    score = confidences
    score = score.permute(1, 2, 3, 0).contiguous()
    score = score.view(2, -1, score.size()[3])
    score = F.softmax(score, dim=0)
    score = score.data[1, :]
    score = score.cpu()
    score = score.numpy()
    return score
    
def convert_bboxK(delta, anchor):
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1, delta.size()[0])
    delta = delta.data.cpu().numpy()
    
    delta[0, :] = delta[0, :] * anchor[:, :, 2] + anchor[:, :, 0]
    delta[1, :] = delta[1, :] * anchor[:, :, 3] + anchor[:, :, 1]
    delta[2, :] = np.exp(delta[2, :]) * anchor[:, :, 2]
    delta[3, :] = np.exp(delta[3, :]) * anchor[:, :, 3]
    return delta

def check_bbox(point, size, boundary):
    boundary = boundary[::-1]
    pkey = np.where(point < boundary, point, boundary)
    point = np.where(0 < pkey, pkey, 0)
    skey = np.where(size < boundary, size, boundary)
    size = np.where(10 < skey, skey, 10)
    return point, size

def iou(bbox, shapes):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    shapes_tl = shapes[:, :2]
    shapes_br = shapes[:, :2] + shapes[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], shapes_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], shapes_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], shapes_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], shapes_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_shapes = shapes[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_shapes - area_intersection)