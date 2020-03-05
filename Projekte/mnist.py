#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:47:02 2019

@author: pandoora
"""
import keras as ks
import tensorflow as tf

Sequential = tf.keras.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = \
tf.keras.datasets.fashion_mnist.load_data()


model = Sequential()# Must define the input shape in the first layer of the neural network
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))# Take a look at the model summary
model.summary()



model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])



x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


x_train  = x_train.reshape(x_train.shape + (1,) )
x_test  = x_test.reshape(  x_test.shape + (1,) )

model.fit(x_train,
         y_train,
         batch_size=100,
         epochs=10,
         validation_data=(x_test, y_test),
         callbacks=[ ModelCheckpoint(
               filepath="weights",
               verbose=1, save_best_only=True)])


model.predict(x_train)
      