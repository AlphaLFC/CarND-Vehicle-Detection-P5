# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:15:04 2017

@author: super
"""

import numpy as np
import cv2


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imgcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imgcopy, bbox[0], bbox[1], color, thick)
    return imgcopy

def draw_points(img, points, color=(255, 0, 0), thick=6):
    imgcopy = np.copy(img)
    for point in points:
        point = tuple(point)
        cv2.line(imgcopy, point, point, color, thick)
    return imgcopy