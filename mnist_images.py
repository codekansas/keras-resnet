"""Generates images at various layers.

To actually train the model, run mnist.py.
You should also make a directory called conv_images (will throw error otherwise).
"""
from __future__ import print_function

import PIL.Image as Image
from pylearn2.utils.image import tile_raster_images

from keras.layers import merge
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from resnet import Residual

folder = 'conv_images'

batch_size = 128
nb_classes = 10
nb_epoch = 3

img_rows, img_cols = 28, 28
pool_size = (2, 2)
kernel_size = (3, 3)

(_, _), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255

# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Model
from keras.models import load_model, Model
model = load_model('mnist_model.h5', custom_objects={'Residual': Residual})

input_layer = model.get_layer('input_1')
input_tensor = input_layer.inbound_nodes[-1].output_tensors[0]

import numpy as np
np.random.seed(42)  # @UndefinedVariable
idx = np.random.randint(2, 100)  # @UndefinedVariable

def save_layer_output(layer, name, shape):
    tensor = layer.inbound_nodes[-1].output_tensors[0]
    model = Model(input_tensor, tensor)
    model.compile(optimizer='sgd', loss='mse')
    pred = model.predict(X_test[idx-1:idx])  # @UndefinedVariable
    reshaped = pred.reshape(*shape)
    tiled = tile_raster_images(X=reshaped.T, img_shape=shape[:-1], tile_shape=(2, shape[-1]/2), tile_spacing=(1, 1))
    image = Image.fromarray(tiled)
    image.save('%s/%s' % (folder, name))

def save_weights(layer, name, shape):
    weights = layer.get_weights()[0].reshape(*shape)
    tiled = tile_raster_images(X=weights.T, img_shape=shape[:-1], tile_shape=(4, 4), tile_spacing=(1, 1))
    image = Image.fromarray(tiled)
    image.save('%s/%s' % (folder, name))

for i in range(2, 6):
    # get only the residual
    layer1 = model.get_layer('residual_%d' % (i - 1))
    layer2 = model.get_layer('residual_%d' % i)
    l1_tensor = layer1.inbound_nodes[-1].output_tensors[0]
    l2_tensor = layer2.inbound_nodes[-1].output_tensors[0]
    diff = merge([l2_tensor, l1_tensor], mode=lambda x: x[1] - x[0], output_shape=lambda x: x[0])
    diff_model = Model(input_tensor, diff)
    diff_model.compile(optimizer='sgd', loss='mse')
    pred = diff_model.predict(X_test[idx-1:idx])
    reshaped = pred.reshape(28, 28, 8)
    tiled = tile_raster_images(reshaped.T, (28, 28), (2, 4), (1, 1))
    image = Image.fromarray(tiled)
    image.save('%s/residual_at_layer_%d.png' % (folder, i))

    # save the filters
    save_weights(model.get_layer('residual_%d' % i), 'filters_at_layer_%d.png' % i, (3, 3, 64))

    # get the output before applying the activation function
    save_layer_output(model.get_layer('residual_%d' % i), 'preactivation_at_layer_%d.png' % i, (28, 28, 8))

    # get the output after applying the actvation function
    save_layer_output(model.get_layer('activation_%d' % i), 'activation_at_layer_%d.png' % i, (28, 28, 8))

