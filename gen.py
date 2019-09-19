#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 04:50:34 2019

@author: pandoora
"""

from skimage.io import imread

def get_input(path):
    
    img = imread(path)
    
    return(img)
    
path = "original/1.png"


images = get_input(path)    



import numpy as np
import pandas as pd

def get_output(path,label_file=None):
    
    img_id = path.split('/')[-1].split('.')[0]
    img_id = np.int64(img_id)
    labels = label_file.loc[img_id].values
    
    return(labels)