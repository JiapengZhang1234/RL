# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:05:44 2020

@author: Administrator
"""

import numpy as np
import cv2


def imgbuffer_process(imgbuffer,out_shape=(84,84)):
    img_list = []
    for img in imgbuffer:
        #将图片reshape为（84,84,3）
        tmp = cv2.resize(src=img, dsize=out_shape)
        #将图像化为（84,84,1）的灰度图
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        #数据归一化
        tmp = cv2.normalize(tmp,tmp,alpha=0.0,beta=1.0,
                           norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        #扩充一个维度
        tmp = np.expand_dims(tmp,len(tmp.shape))
        img_list.append(tmp)
    ret = np.concatenate(tuple(img_list), axis=2)
    return ret
