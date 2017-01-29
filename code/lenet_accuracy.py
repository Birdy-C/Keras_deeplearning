# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:21:43 2016
check accuracy

@author: Cherry

"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
#import matplotlib.cm as cm
#from keras.models import load_model

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten #,Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
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

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# to suit for different kind of image
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Type Cast & normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


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



y_hat = model.predict_classes(X_test)
test_all=[]
test_wrong=[]
all_count=np.zeros((1,10))
right_count=np.zeros((1,10))
for im in zip(X_test,y_hat,y_test):
    test_all=test_all + [tuple(im)]
    if im[1] != im[2]:
        test_wrong = test_wrong+[tuple(im)]
    else:  
        right_count[0,im[2]]= right_count[0,im[2]]+1;
    all_count[0,im[2]]=all_count[0,im[2]]+1;
            
          
def plt_fuc(test):
    plt.figure(figsize=(10, 20))
    for ind, val in enumerate(test[:200]):
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.subplot(10, 20, ind + 1)
        im = 1 - val[0].reshape((28,28))
        plt.axis("off")
        plt.text(0, 0, val[2], fontsize=14, color='blue')
        plt.text(8, 0, val[1], fontsize=14, color='red')
        plt.imshow(im, cmap='gray')

    plt.show()

plt_fuc(test_wrong)
plt_fuc(test_all)

for i in xrange(10):
    print ('number'+ str(i)+ ': '+ str( right_count[0,i] / all_count[0,i]))