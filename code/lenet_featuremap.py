from __future__ import absolute_import
from __future__ import print_function
import pylab as pl
import matplotlib.cm as cm
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#import theano

batch_size = 128
nb_classes = 10
nb_epoch = 1

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
t = X_test[1]
# to suit for different kind of image
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
#we will show the output of the first conv layer
convout1=Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape)
model.add(convout1) 
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Activation("sigmoid"))

#we will show the output of the second conv layer
convout2 = Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1], 
						border_mode='valid')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("sigmoid"))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters[2], 4, 4, border_mode='valid'))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation("sigmoid"))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#load the trained model
model.load_weights('lenet_model_iter_24.h5')

#====================Convolution visualizations===============================#
# K.learning_phase() is a flag that indicates if the network is in training or
# predict phase. It allow layer (e.g. Dropout) to only be applied during training
inputs = [K.learning_phase()] + model.inputs

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
	
# random choose a picture
i = 1200
X = X_test[i:i+1]
print(X)
n=model.predict_classes(X)
pl.figure()
pl.title('input')
nice_imshow(pl.gca(), np.squeeze(X), cmap=cm.binary)
pl.show()
print("prediction:",n)

import numpy.ma as ma
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#======== Visualize the first layer of convolutions on an input image===========#
#model.layers[0] means the first conv
W = model.layers[0].W.get_value(borrow=True)
W = np.squeeze(W)
print("First conv layer shape : ", W.shape)

pl.figure(figsize=(15, 15))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W, 1, 6), cmap=cm.binary)
pl.show()

_convout1_f = K.function(inputs, [convout1.output])
def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

C1 = convout1_f(X)
C1 = np.squeeze(C1)
print("Frist conv layer out shape : ", C1.shape)

pl.figure(figsize=(15, 15))
pl.suptitle('convout1')
nice_imshow(pl.gca(), make_mosaic(C1, 1, 6), cmap=cm.binary)
pl.show()

#======== Visualize the second layer of convolutions on an input image===========#
W2 = model.layers[3].W.get_value(borrow=True)
W2 = np.squeeze(W2)
print("Second conv layer shape : ", W2.shape)

_convout2_f = K.function(inputs, [convout2.output])
def convout2_f(X):
    # The [0] is to disable the training phase flag
    return _convout2_f([0] + [X])

C2 = convout2_f(X)
C2 = np.squeeze(C2)
print("Second conv layer out shape : ", C2.shape)

pl.figure(figsize=(15, 15))
pl.suptitle('convout2')
nice_imshow(pl.gca(), make_mosaic(C2, 4, 4), cmap=cm.binary)
pl.show()
