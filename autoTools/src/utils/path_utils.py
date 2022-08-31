#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Project: STracker
Author: Scheaven
Date: 2022-06-29 18:30:42
LastEditors: Scheaven
LastEditTime: 2022-07-28 15:10:49
Remark: Never cease to pursue!
'''
import os

def get_fatherPath(path):
    path_list = path.split(os.path.sep)
    # print("save path: ",path_list[-2])
    return path.replace(path_list[-2]+"/","")
