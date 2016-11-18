# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:57:49 2016

@author: Administrator
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import MyBow
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation

#生成图片路径
def path(cls,i):
    return '%s\%s-%d.pgm' % (datapath,cls,i)

#特征识别方法与match方法
def create_SiftAndFlann():
    flann_params=dict(algorithm=1,trees=5)
    search_params=dict(checks=50)
    flann=cv2.FlannBasedMatcher(flann_params,search_params)
    detect=cv2.SIFT()
    extract=cv2.SIFT()
    return flann,detect,extract


def extract_sift(fn,extract,detect):
    im=cv2.imread(fn,0)
    return extract.compute(im,detect.detect(im))[1]

#用BOW，Kmeans距离Computer visual BOW.k是聚类的个数，n是正负样本数。
def generate_voc(k,extract,detect,n=20):
    bow_kmeans_trainer=cv2.BOWKMeansTrainer(k)
    for i in range(n):
        des1=extract_sift(path('pos',i),extract,detect)
        des2=extract_sift(path('neg',i),extract,detect)
        if des1 is not None:
            bow_kmeans_trainer.add(des1)
        if des2 is not None:
            bow_kmeans_trainer.add(des2)
    return bow_kmeans_trainer.cluster()
    
    
def extract_bow(voc,extract, flann):
    extract_bow = MyBow.BOWImgDescriptorExtractor(extract, flann)
    extract_bow.setVocabulary(voc) 
    return extract_bow

#建立向量 
def bow_features(fn,extract_bow):
    im=cv2.imread(fn,0)
    return extract_bow.compute(im,detect.detect(im))

#建立训练数据。n是正负样本数。
def get_data(extract_bow,n=40):
    traindata,trainlabels=[],[]
    for i in range(n):
        traindata.extend(bow_features(path('pos',i),extract_bow))
        trainlabels.append(1)
        traindata.extend(bow_features(path('neg',i),extract_bow))
        trainlabels.append(-1)
    return traindata,trainlabels

#交叉验证算法，检查精确度。
if __name__=='__main__':
    datapath=r'.\CarData\CarData\TrainImages'
    flann,detect,extract=create_SiftAndFlann()
    voc=generate_voc(200,extract,detect,n=500)
    extract_bow=extract_bow(voc,extract,flann)
    traindata,trainlabels=get_data(extract_bow,n=500)
    traindata=np.asarray(traindata)    
    trainlabels=np.asarray(trainlabels)   
    clf=Pipeline([('preproc',StandardScaler()),('classifier',SVC())])
    cv=cross_validation.LeaveOneOut(len(traindata))
    scores=cross_validation.cross_val_score(clf,traindata,trainlabels,cv=cv)
    print('Accuracy:{:.1%}'.format(scores.mean()))
#