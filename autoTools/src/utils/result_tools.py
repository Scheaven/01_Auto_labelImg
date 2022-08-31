'''
Project: STracker
Author: Scheaven
Date: 2022-08-01 20:36:34
LastEditors: Scheaven
LastEditTime: 2022-08-01 20:36:39
Remark: Never cease to pursue!
'''
def addResult(label, bndbox, difficult, truncated):
    xmin = int(float(bndbox[0]))
    ymin = int(float(bndbox[1]))
    xmax = int(float(bndbox[0]+bndbox[2]))
    ymax = int(float(bndbox[1]+bndbox[3]))
    points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return (label, points, None, None, difficult, truncated)