# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:01:48 2017

@author: super
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog


class FeatureExtractor(object):
    
    def __init__(self, cfg=None):
        # Define color mode dict.
        self.cdict = {'HSV': cv2.COLOR_RGB2HSV,
                      'HLS': cv2.COLOR_RGB2HLS,
                      'LUV': cv2.COLOR_RGB2LUV,
                      'YUV': cv2.COLOR_RGB2YUV,
                      'YCrCb': cv2.COLOR_RGB2YCrCb}
        if cfg is None:
            self.color_space = 'YCrCb'
            self.orient = 9
            self.pix_per_cell = (8, 8)
            self.cell_per_block = (2, 2)
            self.hog_channel = 'ALL'
            self.spatial_size = (32, 32)
            self.hist_bins = 32
        else:
            self.color_space = cfg.color_space
            self.orient = cfg.orient
            self.pix_per_cell = cfg.pix_per_cell
            self.cell_per_block = cfg.cell_per_block
            self.hog_channel = cfg.hog_channel
            self.spatial_size = cfg.spatial_size
            self.hist_bins = cfg.hist_bins
    
    def extract_hog_feat(self, img):
        '''Extract hog features from a 2d image (one channel).'''
        features = hog(img, orientations=self.orient,
                       pixels_per_cell=self.pix_per_cell,
                       cells_per_block=self.cell_per_block)
        return features
    
    def vis_hog(self, img):
        '''Visualize the hog image.'''
        if self.color_space in self.cdict:
            feat_img = cv2.cvtColor(img, self.cdict[self.color_space])
        _, hog_img = hog(feat_img[:, :, 0], orientations=self.orient,
                         pixels_per_cell=self.pix_per_cell,
                         cells_per_block=self.cell_per_block,
                         visualise=True)
        return hog_img
    
    def extract_bin_spatial(self, img):
        '''Resize an image and flatten it.'''
        return cv2.resize(img, self.spatial_size).ravel()
    
    def extract_color_hist(self, img):
        '''Extract channel histograms from an image.'''
        ch1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins)[0]
        ch2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins)[0]
        ch3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins)[0]
        return np.concatenate([ch1_hist, ch2_hist, ch3_hist])
    
    def extract_features(self, img, do_spatial=True,
                         do_hog=True, do_hist=True):
        '''Extract features of an image by relavent flags.'''
        # Convert color space.
        if self.color_space in self.cdict:
            feat_img = cv2.cvtColor(img, self.cdict[self.color_space])
        else: feat_img = img.copy()
        features = []
        # Extract features by flags.
        if do_spatial:
            spatial_features = self.extract_bin_spatial(feat_img)
            features.append(spatial_features)
        if do_hist:
            hist_features = self.extract_color_hist(feat_img)
            features.append(hist_features)
        if do_hog:
            if self.hog_channel == 'ALL':
                hog_features = []
                for ch in range(feat_img.shape[2]):
                    hog_features.extend(self.extract_hog_feat(
                        feat_img[:, :, ch]))
            else:
                hog_features = self.extract_hog_feat(
                    feat_img[:, :, self.hog_channel])
            features.append(hog_features)
        return np.concatenate(features)
    
    def extract_features_from_imgs(self, imgs, path=True, do_spatial=True,
                                   do_hog=True, do_hist=True):
        all_features = []
        for img in imgs:
            if path:
                img = plt.imread(img, format='RGB')
            features = self.extract_features(img,
                do_spatial=do_spatial, do_hog=do_hog, do_hist=do_hist)
            all_features.append(features)
        return all_features