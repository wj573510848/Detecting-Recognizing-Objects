# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:57:49 2016

@author: Administrator
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
import MyBow

datapath=r'E:\opencv_test\chapter7\CarData\CarData\TrainImages'

def path(cls,i):
    return '%s\%s%d.pgm' % (datapath,cls,i)

 #定义detect函数，match函数，使用BOW建立图向量
 #训练SVM分类
   
flann_params=dict(algorithm=1,trees=5)
search_params=dict(checks=50)
flann=cv2.FlannBasedMatcher(flann_params,search_params)
detect=cv2.SIFT()
extract=cv2.SIFT()
    
pos,neg='pos-','neg-'

#将descriptor聚类为100
bow_kmeans_trainer=cv2.BOWKMeansTrainer(40)


def extract_sift(fn):
    im=cv2.imread(fn,0)
    return extract.compute(im,detect.detect(im))[1]
    
for i in range(20):
    print i
    des1,des2=extract_sift(path(pos,i)),extract_sift(path(neg,i))
    if des1 is not None:
        bow_kmeans_trainer.add(des1)
    if des2 is not None:
        bow_kmeans_trainer.add(des2)
    
#获得图像词袋（BOW）
voc=bow_kmeans_trainer.cluster()
    
#用获得的BOW，建立具体图片的词向量。
extract_bow = MyBow.BOWImgDescriptorExtractor(extract, flann)
extract_bow.setVocabulary(voc) 

#建立图向量 
def bow_features(fn):
    im=cv2.imread(fn,0)
    return extract_bow.compute(im,detect.detect(im))

traindata,trainlabels=[],[]
for i in range(40):
    traindata.extend(bow_features(path(pos,i)))
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg,i)))
    trainlabels.append(-1)
    
svm=cv2.SVM()
svm.train(np.array(traindata,dtype=np.float32),np.array(trainlabels))
#%%
def predict(fn):
    f=bow_features(fn)
    f=f.astype(np.float32)
    p=svm.predict(f)
    return p
