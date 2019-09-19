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

import warnings
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


def classifier():
    model = Sequential()
    model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=["acc"])
    return model

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#model1 =   classifier()
#model1.fit(X, y, epochs=10, batch_size=20,verbose=1)

model = KerasClassifier(build_fn=classifier, epochs=10, batch_size=20, verbose=0)
#
def keras_model(i, k):
    kfold = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=2)
    results = cross_val_score(model, X, y, cv=kfold, n_jobs=-1)
    return np.mean(results)


from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import time, random  

import warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#config = tf.ConfigProto()
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
#sess = tf.Session(config=config)

def eval_network(n1, n2, dropout1,dropout2, batchsize, m):
    
    k = 10
   
    boot = StratifiedShuffleSplit(n_splits=int(k), test_size = 1 /int(k))

    num_cores = multiprocessing.cpu_count()
    rfc_score = np.array(Parallel(n_jobs=num_cores)(delayed(processInput)(train, test, n1, n2,dropout1,dropout2,batchsize, m) for train,test in boot.split(X, y)))

    return np.mean(rfc_score)    


def processInput(train, test, n1, n2, dropout1,dropout2,batchsize, q):
    X_train = X[train]
    y_train = y[train]
    
    X_test  = X[test]
    y_test  = y[test]   
    
    
    model = Sequential()
    model.add(Dense(int(n1), input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout1))
    model.add(Dense(int(n2), activation='relu'))
    model.add(Dropout(dropout2))   
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=["acc"])
    
    
    history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=1, batch_size=int(batchsize),verbose=0)
    
#    if int(m) == 1:
#    
#        knn = KNeighborsClassifier(n)       
#        knn.fit(X_train, y_train)
#        
#        score = knn.score(X_test, y_test)
#    
#    elif it(m) == 2:
#        rfc = RandomForestClassifier(i)
#        rfc.fit(X_train, y_train)
#        
#        score = rfc.score(X_test, y_test)
        
    return history.history['val_acc'][-1]


# Bounded region of parameter space
pbounds = {'n1': (1, 100), 'n2': (1, 100), 'n2': (1, 100), 'dropout1': (0.1, 0.5), 'dropout2': (0.1, 0.5),  'batchsize': (1, 100)}

optimizer = BayesianOptimization(
    f=eval_network,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=15,
    n_iter=2,
)

print(optimizer.max)



