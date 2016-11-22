#!/usr/bin/env python

import numpy as np
import pygast
import cv2
from skimage.feature import peak_local_max

class GASTDetector(object):
    def __init__(self, levels=3, sigma=7, adjust_contrast=True, grid_size=(8,8)):
        self.levels = levels
        self.sigma = sigma
        self.adjust_contrast = adjust_contrast
        self.grid_size = grid_size
        if self.adjust_contrast:
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=self.grid_size)
        self.syms = [pygast.Symmetry() for i in xrange(levels)]
        
    def get_pyramid(self, g):
        imgs = []
        imgs.append(g)
        for i in xrange(1, self.levels):
            imgs.append(cv2.pyrDown(imgs[i-1]))
        return imgs

    def detect(self, img, min_distance=10):
        if len(img.shape) == 3:
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gimg = img
        if self.adjust_contrast:
            gimg = self.clahe.apply(gimg)
        gpyr = self.get_pyramid(gimg)
        smags = [np.zeros_like(gpyr[i], dtype=np.float32) for i in xrange(self.levels)]
        sdirs = [np.zeros_like(gpyr[i], dtype=np.float32) for i in xrange(self.levels)]
        for i in xrange(self.levels):
            self.syms[i].transform(gpyr[i], smags[i], sdirs[i], self.sigma)
        # combine the scales, after resizing
        composite = smags[0] / np.amax(smags[0])
        full_shape = gimg.shape[::-1]
        for i in xrange(1, self.levels):
            layer = cv2.resize(smags[i], full_shape)
            layer = layer / np.amax(layer)
            composite = composite + layer
        max_coords = peak_local_max(composite, min_distance)
        return max_coords, composite
