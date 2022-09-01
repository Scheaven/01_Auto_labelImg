#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Project: STracker
Author: Scheaven
Date: 2022-06-29 18:30:42
LastEditors: Scheaven
LastEditTime: 2022-09-01 11:08:11
Remark: Never cease to pursue!
'''
import os

def get_fatherPath(path):
    path_list = path.split(os.path.sep)
    return path.replace(path_list[-2]+"/","")
