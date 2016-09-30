"""Train a convnet on the MNIST database with ResNets.

ResNets are a bit overkill for this problem, but this illustrates how to use
the Residual wrapper on ConvNets.

See: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
"""
from __future__ import print_function
import numpy as np
np.random.seed(42)

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from resnet import Residual

batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28
pool_size = (2, 2)
kernel_size = (3, 3)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

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
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Model
input = Input(shape=input_shape)

def get_deep_model(input):
    conv1 = Convolution2D(8, kernel_size[0], kernel_size[1],
                          border_mode='valid', activation='relu')(input)
    resnet = conv1
    for i in range(5):
        resnet = Residual(Convolution2D(8, kernel_size[0], kernel_size[1],
                                      border_mode='same'))(resnet)
        resnet = Activation('relu')(resnet)
    mxpool = MaxPooling2D(pool_size=pool_size)(resnet)
    dropout = Dropout(0.25)(mxpool)
    flat = Flatten()(dropout)
    dense = Dense(128, activation='relu')(flat)
    return dense

def get_wide_model(input):
    conv = Convolution2D(128, kernel_size[0], kernel_size[1],
                         border_mode='valid', activation='relu')(input)
    mxpool = MaxPooling2D(pool_size=pool_size)(conv)
    dropout = Dropout(0.25)(mxpool)
    flat = Flatten()(dropout)
    dense = Dense(128, activation='relu')(flat)
    return dense

merged = merge([get_wide_model(input), get_deep_model(input)], mode='max')
dropout = Dropout(0.5)(merged)
softmax = Dense(nb_classes, activation='softmax')(dropout)

model = Model(input=[input], output=[softmax])
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
