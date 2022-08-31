'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 15:57:05
LastEditors: Scheaven
LastEditTime: 2022-07-29 17:53:50
Remark: Never cease to pursue!
'''
from multiprocessing.connection import wait
from os import path
import time
import cv2
from cv2 import imshow
from cv2 import waitKey
import numpy as np
import torch
from autoTools.src.config import cfg
torch.set_printoptions(profile="full")
class BaseTracker(object):
    def __init__(self) -> None:
        self.ACFalg = True
    def init(self, img, bbox):
        raise NotImplementedError
    
    def track(self, img):
        raise NotImplementedError

    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """  
    def Tcrop_img(self, im, pos, model_sz, original_sz):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[:, 0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[:, 1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        # print(sz , ":" , im_sz , ":" ,c , original_sz)
        # print(context_xmin, context_xmax, context_ymin, context_ymax)
        # print("--------------------------------------")
        left_pad = np.where(0. < context_xmin, 0, (-context_xmin).astype(np.int64))
        top_pad = np.where(0. < context_ymin, 0, (-context_ymin).astype(np.int64))
        right_pad = np.where(0. < (context_xmax - im_sz[1] + 1), (context_xmax - im_sz[1] + 1).astype(np.int64), 0)
        bottom_pad = np.where(0. < (context_ymax - im_sz[0] + 1), (context_ymax - im_sz[0] + 1).astype(np.int64), 0)

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        print(context_xmin, context_xmax, context_ymin, context_ymax)
        print("------------left_pad--------------------------")
        print(left_pad, top_pad, right_pad, bottom_pad)
        # cv2.imwrite("02_img.jpg", im)
        # exit(0)
        # im = cv2.imread("02_img.jpg")

        r, c, k = im.shape
        if self.ACFalg:
            self.avg_chans = np.mean(im, axis=(0, 1))
            self.ACFalg = False
        patch_list = []
        for i in range(context_xmax.size):
            if np.any([top_pad, bottom_pad, left_pad, right_pad]):
                size = (r + top_pad[i] + bottom_pad[i], c + left_pad[i] + right_pad[i], k)
                te_im = np.zeros(size, np.uint8)
                te_im[top_pad[i]:top_pad[i] + r, left_pad[i]:left_pad[i] + c, :] = im
                if top_pad[i]:
                    te_im[0:top_pad[i], left_pad[i]:left_pad[i] + c, :] = self.avg_chans
                if bottom_pad[i]:
                    te_im[r + top_pad[i]:, left_pad[i]:left_pad[i] + c, :] = self.avg_chans
                if left_pad[i]:
                    te_im[:, 0:left_pad[i], :] = self.avg_chans
                if right_pad[i]:
                    te_im[:, c + left_pad[i]:, :] = self.avg_chans
                im_patch = te_im[int(context_ymin[i]):int(context_ymax[i] + 1),
                                int(context_xmin[i]):int(context_xmax[i] + 1), :]
            else:
                im_patch = im[int(context_ymin[i]):int(context_ymax[i] + 1),
                            int(context_xmin[i]):int(context_xmax[i] + 1), :]
 
            # cv2.imwrite("03_img.jpg", im_patch)
            # imsdf = cv2.imread("03_img.jpg")
            # print("imgdfo04:", imsdf.shape)
            # if not np.array_equal(model_sz, original_sz):
            imsdf = cv2.resize(im_patch, (model_sz, model_sz))

            # cv2.imshow("sdf", im_patch)
            # cv2.waitKey(0)
            # print(im_patch)
            # print(im_patch.shape)
            n_im_patch = imsdf
            # n_im_patch = cv2.imread("01_test.jpg")
            # print(":::::::",int(context_ymin[i]),int(context_ymax[i] + 1),
            #                 int(context_xmin[i]),int(context_xmax[i] + 1))
            n_im_patch = n_im_patch.transpose(2, 0, 1).astype(np.float32)
            n_im_patch = torch.from_numpy(n_im_patch)
            if cfg.CUDA:
                n_im_patch = n_im_patch.cuda()
            n_im_patch = n_im_patch.unsqueeze(0)
            # print(n_im_patch)
            # exit(0)
            patch_list.append(n_im_patch)

        patchs = torch.cat(patch_list, dim=0)
        return patchs
