# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:20:34 2017

@author: super
"""

import cv2
import numpy as np

from collections import deque
from scipy.ndimage.measurements import label
from utils import draw_boxes

class VehicleFinder4Image(object):
    
    def __init__(self, windows, clf):
        self.windows = windows
        self.clf = clf
    
    def search_windows(self, img):
        hot_windows = []
        imgs = []
        for window in self.windows:
            (x1, y1), (x2, y2) = window
            if (x2 - x1 != 64) and (y2 - y1 != 64):
                img_win = cv2.resize(img[y1:y2, x1:x2], (64, 64))
            else: img_win = img[y1:y2, x1:x2]
            imgs.append(img_win)
        predictions = self.clf.predict(imgs)
        for i, window in enumerate(self.windows):
            if predictions[i] == 1:
                hot_windows.append(window)
        return hot_windows
    
    def add_heat(self, heatmap, hot_windows):
        for window in hot_windows:
            (x1, y1), (x2, y2) = window
            heatmap[y1:y2, x1:x2] += 1
        return heatmap
    
    def heat_thresh(self, heatmap, thresh):
        heatmap = cv2.resize(heatmap, (80, 45))
        heatmap = cv2.resize(heatmap, (1280, 720))
        heatmap[heatmap <= thresh] = 0
        return heatmap
    
    def detect_vehicles_bboxes(self, heatmap):
        labels = label(heatmap)
        bboxes = []
        for idx in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == idx).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)
        return bboxes
    
    def apply(self, img):
        hot_windows = self.search_windows(img)
        heatmap = np.zeros_like(img[:,:,0], dtype=np.float32)
        heatmap = self.add_heat(heatmap, hot_windows)
        heatmap = self.heat_thresh(heatmap, 4)
        vehicle_bboxes = self.detect_vehicles_bboxes(heatmap)
        img_detected = draw_boxes(img, vehicle_bboxes)
        return img_detected


class VehicleFinder4Video(VehicleFinder4Image):
    
    def __init__(self, windows, clf, n_recs=5):
        super(VehicleFinder4Video, self).__init__(windows, clf)
        self.n_recs = n_recs
        self.heatmaps = deque(maxlen=self.n_recs)
    
    def apply(self, img):
        hot_windows = self.search_windows(img)
        heatmap = np.zeros_like(img[:,:,0], dtype=np.float32)
        heatmap = self.add_heat(heatmap, hot_windows)
        self.heatmaps.append(heatmap)
        heatmap = np.mean(self.heatmaps, axis=0)
        heatmap = self.heat_thresh(heatmap, 4)
        vehicle_bboxes = self.detect_vehicles_bboxes(heatmap)
        img_detected = draw_boxes(img, vehicle_bboxes)
        return img_detected