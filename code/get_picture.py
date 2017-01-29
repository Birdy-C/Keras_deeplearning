# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:58:05 2016

@author: Cherry
"""

from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
#import scipy as sp
#import matplotlib.cm as cm
#from keras.models import load_model
import matplotlib.image as mpimg
#from compiler.ast import flatten

np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
#import pydot
#from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils.visualize_util import plot
#from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
nb_classes = 10
nb_epoch = 24
data_augmentation= True
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = (6, 16, 120)
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)


input_shape = (1, img_rows, img_cols)

model = Sequential()

model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Activation("sigmoid"))

model.add(Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1], 
						border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("sigmoid"))

model.add(Convolution2D(nb_filters[2], 4, 4, border_mode='valid'))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation("sigmoid"))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

plot(model, to_file='lenet.png', show_shapes='true')
#load the trained model
model.load_weights('lenet_mnist_iter_xy.h5')

#get the picture
X=mpimg.imread('number.jpg')
X_train1=255-X[:,:,1]
X_train1 = X_train1.reshape(1,1, 28, 28)
# Type Cast & normalize
X_train1 = X_train1.astype('float32')
X_train1 /= 255
t=model.predict_classes(X_train1)
print("prediction:",t)