#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:35:24 2019

@author: pandoora
"""
import tensorflow as tf
keras = tf.keras
import pandas as pd
import numpy as np


Sequential =  keras.models.Sequential
Dense = keras.layers.Dense
Activation =  keras.layers.Activation
metrics = keras.metrics
Dropout = keras.layers.Dropout
ModelCheckpoint = keras.callbacks.ModelCheckpoint
KerasClassifier = keras.wrappers.scikit_learn.KerasClassifier


import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
#import multiprocessing
#import concurrent.futures
import time, random  

import warnings
seed = 7

warnings.filterwarnings("ignore", category=FutureWarning)

from library import binary_encoding, zscore_normalisation, rounding

einkommen = pd.read_csv("einkommen.train")

# changing index cols with 
einkommen.columns = ['Age', 'Employment', 'Weighting factor', 'Level of education', 
                'Schooling/training period', 'Marital status', 'Employment area', 'Partnership', 
                'Ethnicity', 'Gender', 'Gains on financial assets', 'Losses on financial assets', 
                'Weekly working time', 'Country of birth','Income'] 

#the set with given Target
set_1 = einkommen[:4999]

#check for numbers
int_col = [col for i, col in enumerate(set_1.columns) if set_1[col].dtype == int ]

#z normalization for nummeric values
znorm_df = zscore_normalisation(set_1, int_col)

#check for nominals
nominal_col = [col for i, col in enumerate(znorm_df.columns) if set_1[col].dtype != int ]

print("Check for Missing values in nominal")
for i,col in enumerate(nominal_col):
    print(i,"Number of ? contains in ",col, sum(znorm_df[col].str.contains('?', regex=False))) 

print("nonimal columns ", nominal_col)

#checking for binarys in nominal col
binary_columns = []
for i, col in enumerate(nominal_col):
    unique_names = set_1.loc[:,col].unique()
    if len(unique_names) == 2:
        binary_columns.append(col)

#encoding binary data
bin_encoded, maps = binary_encoding(znorm_df,binary_columns)
print("Dictionary for binary encodings", maps)

#throwing new binary encoded columns
nominal_col = [col for i, col in enumerate(bin_encoded.columns) if bin_encoded[col].dtype != int and bin_encoded[col].dtype != float ]
print(nominal_col)

#one hot encoding the nominals
transformed_df = pd.get_dummies(bin_encoded, columns=nominal_col)

#Get the Target
y = transformed_df["Income"].values
X = transformed_df.drop(columns=['Income']).values
#
#
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.1, random_state=seed)


def eval_kerasmodel(n1,n2, d1, d2):

    def classifier():
        model = Sequential()
        model.add(Dense(int(n1), input_dim=X.shape[1], activation='relu'))
        model.add(Dropout(d1))
        model.add(Dense(int(n2), activation='relu'))
        model.add(Dropout(d2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=["acc"])
        return model
    
    model = KerasClassifier(build_fn=classifier, epochs=10, batch_size=20, verbose=0)
    
    def keras_model(X,y):
        kfold = StratifiedKFold(n_splits=int(10), shuffle=True, random_state=2)
        results = cross_val_score(model, X, y, cv=kfold, n_jobs=-1)
        return np.mean(results)
    
    return keras_model(X_train, y_train)


def eval_random_forest(i):
    
    def Random_forest(train, test, i):
        X_train = X[train]
        y_train = y[train]
        
        X_test  = X[test]
        y_test  = y[test]   
        
        rfc = RandomForestClassifier(int(i))
        rfc.fit(X_train, y_train)
        
        score = rfc.score(X_test, y_test)
    
        return score
   
    boot = StratifiedShuffleSplit(n_splits=int(10), test_size = 1 /int(10))

    rfc_score = np.array(Parallel(n_jobs=-1)(delayed(Random_forest)(train,test,i) for train,test in boot.split(X, y)))

    return np.mean(rfc_score)    


#
#pbounds = {'i': (1, 100)}
##
#optimizer = BayesianOptimization(
#    f=eval_random_forest,
#    pbounds=pbounds,
#    random_state=seed,
#)
#
#optimizer.maximize(
#    init_points=5,
#    n_iter=5,
#)
#
#print(optimizer.max)
    
    

pbounds = {'n1': (1, 10),'n2': (1, 100),'d1': (0, 0.99),'d2': (0, 0.99)}
#
optimizer = BayesianOptimization(
    f=eval_kerasmodel,
    pbounds=pbounds,
    random_state=seed,
)

optimizer.maximize(
    init_points=5,
    n_iter=5,
)

print(optimizer.max)
   

#    
#    if m == 1:
#    
#        knn = KNeighborsClassifier(n)       
#        knn.fit(X_train, y_train)
#        
#        score = knn.score(X_test, y_test)
#    
#    elif m == 2:
#        rfc = RandomForestClassifier(i)
#        rfc.fit(X_train, y_train)
#        
#        score = rfc.score(X_test, y_test)
        

#
## Bounded region of parameter space
#pbounds = {'i': (1, 100), 'k': (2, 20), 'n': (2, 20), 'm': (1, 2)}


