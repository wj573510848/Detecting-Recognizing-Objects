# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:55:37 2016

@author: Administrator
"""

import cv2
import numpy as np

class BOWImgDescriptorExtractor(object):
    def __init__(self,dextractor,matcher):
        self.extractor=dextractor
        self.matcher=matcher
        self._voc=None
       
    def setVocabulary(self,voc):
        self._voc=voc
        
    def compute(self,im,des):
        if self._voc is None:
            print 'Vocabulary is None!'
        else:
            bowArray=np.zeros(len(self._voc))
            imgDes=self.extractor.compute(im, des)[1]
            matches=self.matcher.match(imgDes,self._voc)
            for i in range(len(matches)):
                bowArray[matches[i].trainIdx]+=1
            return bowArray.reshape(1,-1)
                

    