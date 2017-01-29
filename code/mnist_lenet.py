'''
	Trains a lenet on the MNIST dataset.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from keras.models import load_model

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#import pydot
from keras.optimizers import SGD#, Adadelta, Adagrad
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
nb_classes = 10
nb_epoch = 50
#whether use data_augmentation
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

plot(model, to_file='lenet.png', show_shapes='true')

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
else:
    print('Using real-time data augmentation.')
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,metrics=['accuracy'])
    datagen = ImageDataGenerator(              
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False, 
        vertical_flip = False)
    datagen.fit(X_train)                         

    hist = model.fit_generator(datagen.flow(X_train, Y_train,    
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test)) 

model.save_weights('lenet_mnist_iter_xy.h5')

#draw plot about accuracy vs epoch
def figures(history,figure_name="plots"):  
    """ method to visualize accuracies and loss vs epoch for training as well as testind data\n 
        Argumets: history     = an instance returned by model.fit method\n 
                  figure_name = a string representing file name to plots. By default it is set to "plots" \n 
       Usage: hist = model.fit(X,y)\n              figures(hist) """  
    from keras.callbacks import History  
    if isinstance(history,History):  
        hist     = history.history   
        epoch    = history.epoch  
        acc      = hist['acc']  
        loss     = hist['loss']  
        val_loss = hist['val_loss']  
        val_acc  = hist['val_acc']  
        plt.figure(1)  
  
        plt.subplot(221)  
        plt.plot(epoch,acc)  
        plt.title("Training accuracy vs Epoch")  
        plt.xlabel("Epoch")  
        plt.ylabel("Accuracy")       
  
        plt.subplot(222)  
        plt.plot(epoch,loss)  
        plt.title("Training loss vs Epoch")  
        plt.xlabel("Epoch")  
        plt.ylabel("Loss")    
  
        plt.subplot(223)  
        plt.plot(epoch,val_acc)  
        plt.title("Validation Acc vs Epoch")  
        plt.xlabel("Epoch")  
        plt.ylabel("Validation Accuracy")    
  
        plt.subplot(224)  
        plt.plot(epoch,val_loss)  
        plt.title("Validation loss vs Epoch")  
        plt.xlabel("Epoch")  
        plt.ylabel("Validation Loss")    
        plt.tight_layout()  
        plt.savefig(figure_name)  
    else:  
        print ('Input Argument is not an instance of class History')
        
        
figures(hist)