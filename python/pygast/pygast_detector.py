#!/usr/bin/env python

import cv2
import numpy as np
from collections import defaultdict
from skimage.feature import peak_local_max

import pygast
from util import Timer, timeit, Stats



class GASTDetector(object):
    def __init__(self, levels=3, sigma=7, adjust_contrast=True, grid_size=(8,8)):
        self.levels = levels
        self.sigma_ = sigma
        self.adjust_contrast = adjust_contrast
        self.grid_size = grid_size
        if self.adjust_contrast:
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=self.grid_size)
        self.syms = [pygast.Symmetry() for i in xrange(levels)]
        self.stats_ = Stats()

    @property
    def sigma(self):
        return self.sigma_

    @sigma.setter
    def sigma(self, s):
        self.sigma_ = s

    @property
    def stats(self):
        return self.stats_
    
    def get_pyramid(self, g):
        imgs = []
        imgs.append(g)
        for i in xrange(1, self.levels):
            imgs.append(cv2.pyrDown(imgs[i-1]))
        return imgs

    def log(self, desc, elapsed):
        #self.stats_[desc].append(elapsed)
        self.stats_.metric(desc, elapsed)

    def cpu_name(self, pyr, smag, sdir, sigma):
        shp = pyr.shape
        shpstr = "{0}x{1}".format(shp[1], shp[0])
        return ["CPU", shpstr, sigma]
        #return "CPU Image {0}x{1}, Radius {2}".format(shp[1],shp[0],sigma)

    def gpu_name(self, pyr, smag, sdir, sigma):
        shp = pyr.shape
        shpstr = "{0}x{1}".format(shp[1], shp[0])
        return ["GPU", shpstr, sigma]
        #return "GPU Image {0}x{1}, Radius {2}".format(shp[1],shp[0],sigma)

    def _detect(self, fn, img, min_distance):
        if len(img.shape) == 3:
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gimg = img.clone()
        if self.adjust_contrast:
            gimg = self.clahe.apply(gimg)
        gpyr = self.get_pyramid(gimg)
        smags = [np.zeros_like(gpyr[i], dtype=np.float32) for i in xrange(self.levels)]
        sdirs = [np.zeros_like(gpyr[i], dtype=np.float32) for i in xrange(self.levels)]
        for i in xrange(self.levels):
            fn[i](gpyr[i], smags[i], sdirs[i], self.sigma_)
            #self.syms[i].transform(gpyr[i], smags[i], sdirs[i], self.sigma)
        # combine the scales, after resizing
        composite = smags[0] / np.amax(smags[0])
        full_shape = gimg.shape[::-1]
        for i in xrange(1, self.levels):
            layer = cv2.resize(smags[i], full_shape)
            layer = layer / np.amax(layer)
            composite = composite + layer
        max_coords = peak_local_max(composite, min_distance)
        return max_coords, composite
    
    def cpu_detect(self, img, min_distance=10):
        fns = [timeit(self.syms[i].cpu_transform,
                      logger=self.log, name_fn=self.cpu_name) for i in xrange(self.levels)]
        return self._detect(fns, img, min_distance)
    
    def detect(self, img, min_distance=10):
        fns = [timeit(self.syms[i].transform,
                      logger=self.log, name_fn=self.gpu_name) for i in xrange(self.levels)]
        return self._detect(fns, img, min_distance)
