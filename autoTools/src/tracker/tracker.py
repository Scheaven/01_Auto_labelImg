'''
Project: STracker
Author: Scheaven
Date: 2022-01-12 15:51:36
LastEditors: Scheaven
LastEditTime: 2022-09-01 11:04:23
Remark: Never cease to pursue!
'''
from cv2 import threshold
import numpy as np
import torch,time
import torch.nn.functional as F
import cv2
from autoTools.src.config import cfg
from autoTools.src.tracker.anchor import Anchors
from autoTools.src.models.builder import ModelBuilder
from autoTools.src.tracker.base_tracker import BaseTracker
from autoTools.src.utils.model_tools import *
from autoTools.src.models.model_load import *
import sys
np.set_printoptions(threshold=sys.maxsize)
# torch.set_printoptions(profile="full")

class Tracker(BaseTracker):
    def __init__(self):
        super().__init__()
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        
        model = ModelBuilder()
        model = self.load_pretrain(model, "autoTools/label_model.pth").cuda().eval().to(device)
        model.eval()

        self.model = model
        self.init_paras()
        print("------------model init ok-----------")
    def reset_autoModel(self):
        self.init_paras()

    def init_paras(self):
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.score_size = 21
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)  #外积
        self.window = np.tile(window.flatten(), self.anchor_num)  # copy self.anchor_num
        self.anchors = self.generate_anchor(self.score_size)

    def load_pretrain(self, model, pretrained_path):
        logger.info('load pretrained model from {}'.format(pretrained_path))
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path,
            map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                            'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')

        try:
            check_keys(model, pretrained_dict)
        except:
            logger.info('[Warning]: using pretrain as features.\
                    Adding "features." as prefix')
            new_dict = {}
            for k, v in pretrained_dict.items():
                k = 'features.' + k
                new_dict[k] = v
            pretrained_dict = new_dict
            check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def set_inf(self, img, bboxs):
        self.bboxs_center = np.array([bboxs[:, 0]+(bboxs[:, 2]-1)/2, bboxs[:, 1]+(bboxs[:, 3]-1)/2]).T
        self.bboxs_size = bboxs[:, 2:4]
        w_z = self.bboxs_size.T + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.bboxs_size, axis=1) # 按比例裁剪
        s_z = np.round(np.sqrt(w_z[0, :]*w_z[1, :]))

        z_crop = self.Tcrop_img(img, self.bboxs_center,
                            cfg.TRACK.EXEMPLAR_SIZE,
                            s_z)
        return z_crop

    def init(self, img, bboxs):
        batchInputs = self.set_inf(img, np.array(bboxs))

        # 生成script pt
        prob_outs = self.model.template(batchInputs)
        self.window = np.tile(self.window, (len(bboxs), 1)).transpose(1, 0)
        self.anchors = np.tile(self.anchors[:, np.newaxis, :], (1, len(bboxs), 1))

    def currCrop(self, img):
        w_z = self.bboxs_size.T + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.bboxs_size, axis=1)  # 按比例裁剪
        s_z = np.round(np.sqrt(w_z[0, :]*w_z[1, :]))

        self.scale = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.Tcrop_img(img, self.bboxs_center,
                                cfg.TRACK.INSTANCE_SIZE,
                                s_x)
        return x_crop

    def infer(self, img):
        batchInputs = self.currCrop(img)
        outputs = self.model.track(batchInputs)
        score = convert_confK(outputs['cls'])
        pred_bbox = convert_bboxK(outputs['loc'], self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :, :], pred_bbox[3, :, :]) /
                     (sz(self.bboxs_size[:, 0]*self.scale, self.bboxs_size[:, 1]*self.scale)))

        r_c = change((self.bboxs_size[:, 0]/self.bboxs_size[:, 1]) /
                     (pred_bbox[2, :, :]/pred_bbox[3, :, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore, axis=0)

        bbox_list = []
        score_list = []
        penalty_list = []
        for i in range(best_idx.size):
            bbox_list.append((pred_bbox[:, best_idx[i], i]/self.scale[i])[np.newaxis, :])
            score_list.append(score[best_idx[i], i])
            penalty_list.append(penalty[best_idx[i], i])

        bbox = np.concatenate(bbox_list, axis=0)
        bbox_score = np.array(score_list)
        bbox_penalty = np.array(penalty_list)
        lr = np.tile(bbox_penalty * bbox_score * cfg.TRACK.LR, (2, 1)).T

        cpoint = bbox[:, 0:2] + self.bboxs_center[:, 0:2]
        n_size = self.bboxs_size * (1 - lr) + bbox[:, 2:4] * lr

        self.bboxs_center, self.bboxs_size = check_bbox(cpoint, n_size, img.shape[:2])
        bbox = np.concatenate(((self.bboxs_center - self.bboxs_size/2), self.bboxs_size), axis=1)
        return {
            'bbox': bbox.tolist(),   # [x, y, width, height]
            'best_score': bbox_score.tolist()
        }
