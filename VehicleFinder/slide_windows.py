# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:13:28 2017

@author: super
"""

import numpy as np
import cv2


class WarpPerspective(object):

    def __init__(self, src=None, dst=None, img_size=(1280, 720)):
        self.img_size = img_size
        if src is None or dst is None:
            self.src = np.float32(
                [[(img_size[0] / 2) - 68, img_size[1] / 2 + 90],
                 [0, img_size[1]],
                 [img_size[0], img_size[1]],
                 [(img_size[0] / 2) + 62, img_size[1] / 2 + 90]])
            self.dst = np.float32(
                [[img_size[0] * 2/5, img_size[1] / 4],
                 [img_size[0] * 2/5, img_size[1]],
                 [img_size[0] * 3/5, img_size[1]],
                 [img_size[0] * 3/5, img_size[1] / 4]])
        else:
            self.src = np.float32(src)
            self.dst = np.float32(dst)
        # Calculate transform matrix and inverse matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        '''Warp an image from perspective view to bird view.'''
        assert (img.shape[1], img.shape[0]) == self.img_size, 'Invalid image shape.'
        return cv2.warpPerspective(img, self.M, self.img_size)

    def warp_inv(self, img):
        '''Warp inversely an image from bird view to perspective view.'''
        assert (img.shape[1], img.shape[0]) == self.img_size, 'Invalid image shape.'
        return cv2.warpPerspective(img, self.M_inv, self.img_size)


class SlidingWindows(object):
    
    def __init__(self, wp, imgsize=(1280, 720), sep=[25, 25]):
        self.wp = wp
        self.imgsize = imgsize
        self.sep = sep  # grid point separation
        self._make_grid_points()
        self._make_sliding_windows()
    
    def _make_grid_points(self):
        '''Generate grid points in a bird view.'''
        xsize, ysize = self.imgsize
        self._xsteps = int(xsize / self.sep[0]) + 1
        self._ysteps = int(ysize / self.sep[1])
        grid = np.mgrid[0:self._xsteps, 0:self._ysteps]
        self.grid_points = grid.T.reshape((-1, 2)) * self.sep
        # Transform the grid points in a perspective view.
        points_transformed = []
        for point in self.grid_points:
            coord = np.append(point, 1)
            transformed = np.dot(self.wp.M_inv, coord)
            point_transformed = (transformed[:2] / transformed[2]).astype(np.int)
            points_transformed.append(point_transformed)
        self.points_transformed = np.array(points_transformed)
        
    def _select_ysteps(self, ysteps):
        yseq = np.arange(ysteps - 1)
        yseq_r = yseq[::-1]  # np.flipud(yseq)
        n = np.int(np.sqrt(4 * ysteps + 0.25) + 0.5)
        idx_list = []
        for i in range(n-1):
            tmp = np.int(i * (i+1) / 4)
            idx_list.append(tmp)
        idx_list = np.unique(idx_list)
        return np.unique(np.append(yseq_r[idx_list], 0)[::-1])
    
    def _make_sliding_windows(self):
        window_list = []
        xseq = range(self._xsteps)
        yseq = self._select_ysteps(self._ysteps)
        for yc in yseq:
            xl = self.points_transformed[yc*self._xsteps][0]
            xr = self.points_transformed[(yc+1)*self._xsteps-1][0]
            width = int( (50/self.sep[0]) * (xr-xl) / (self._xsteps-1) )
            for xc in xseq:
                pos = yc*self._xsteps + xc
                point = self.points_transformed[pos]
                lu = point - np.array([0.5*width, 1.5*width], np.int).tolist()
                rd = point + np.array([1.5*width, 0.5*width], np.int).tolist()
                if 0 < lu[0] < self.imgsize[0] and rd[1] < self.imgsize[1]:
                    window_list.append((tuple(lu), tuple(rd)))
        self.windows = window_list



def slide_window(imgsize=(1280, 720), xrange=[None, None], yrange=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if xrange[0] == None:
        xrange[0] = 0
    if xrange[1] == None:
        xrange[1] = imgsize[0]
    if yrange[0] == None:
        yrange[0] = 0
    if yrange[1] == None:
        yrange[1] = imgsize[1] 
    xspan = xrange[1] - xrange[0]
    yspan = yrange[1] - yrange[0]
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer) / nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer) / ny_pix_per_step) 
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + xrange[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + yrange[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list