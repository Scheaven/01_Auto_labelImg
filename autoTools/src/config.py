'''
Project: STracker
Author: Scheaven
Date: 2022-01-19 17:57:09
LastEditors: Scheaven
LastEditTime: 2022-05-12 21:50:54
Remark: Never cease to pursue!
'''
# -*- coding: utf-8 -*-
# @Author: Scheaven
# @Date:   2022-01-12 23:02:44
# @Last Modified time: 2022-01-13 22:42:45
from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.CUDA = True

# # ------------------------------------------------------------------------ #
# # Anchor options
# # ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = 5


# # ------------------------------------------------------------------------ #
# # Tracker options
# # ------------------------------------------------------------------------ #
__C.TRACK = CN()

# __C.TRACK.TYPE = 'SiamRPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.16

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.40

# Interpolation learning rate
__C.TRACK.LR = 0.3

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 287

# Base size
__C.TRACK.BASE_SIZE = 4

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

