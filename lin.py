#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:06:24 2019

@author: pandoora
"""

import urllib
import pandas as pd
import numpy as np
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

testfile = urllib.request.URLopener()
testfile.retrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train", "SPECT.train")
testfile.retrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test", "SPECT.test")

df_train = pd.read_csv('SPECT.train',header=None)
df_test = pd.read_csv('SPECT.test',header=None)

train = df_train.as_matrix()
test = df_test.as_matrix()

y_train = train[:,0]
X_train = train[:,1:]
y_test = test[:,0]
X_test = test[:,1:]

def loss(h, y):
    #hinge loss vector
    l  = list(map(lambda x: 1-x[0]*x[1] if (1-x[0]*x[1]) > 0  else 0 , zip(y,h)))
    #get gradient
    g = list(map(lambda x: -x[0]*x[1] if (1-x[0]*x[1]) > 0  else 0 , zip(y,h)))
    
    return l, g

def reg(w, lbda):
    r = lbda/2 *np.dot(w.T,w)
    g = lbda * w
    return r, g

def learn_reg_ERM(X,y,lbda):
    #max iteration do go our loop regression training
    max_iter = 200
    #learning rate
    e  = 0.001
    alpha = 1.
    #generate 22 the coeffs for each input Value X
    w = np.random.randn(X.shape[1]);
    for k in np.arange(max_iter):
        #inner product for each input with the coeffs 
        # X%*%v
        h = np.dot(X,w)
        #get the loss between prediction and label 
        l,lg = loss(h, y)
        #print the loss of the current learning model
        print ('loss: {}'.format(np.mean(l)))
        #compute the regulazations and the gradient of the regularizer
        r,rg = reg(w, lbda)
        g = np.dot(X.T,lg) + rg 
        if (k > 0):
            alpha = alpha * (np.dot(g_old.T,g_old))/(np.dot((g_old - g).T,g_old))
        w = w - alpha * g
        if (np.linalg.norm(alpha * g) < e):
            break
        g_old = g
    #return the trained weights    
    return w


w =  learn_reg_ERM(X_train,y_train,1)

