"""Generate images at various layers"""
from __future__ import print_function
import numpy as np


try:
    import PIL.Image as Image
except ImportError:
    import Image
from pylearn2.utils.image import tile_raster_images

from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from resnet import Residual

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

for i in range(1, 6):
    conv = model.get_layer('residual_%d' % i)
    weights = conv.get_weights()[0].reshape(3, 3, 8*8)
    tiled = tile_raster_images(X=weights.T, img_shape=(3, 3), tile_shape=(8, 8), tile_spacing=(1, 1))
    image = Image.fromarray(tiled)
    image.save('residual_images/activations_of_layer_%d.png' % i)
    
    layer = model.get_layer('activation_%d' % i)
    tensor = layer.inbound_nodes[-1].output_tensors[0]
    m = Model(input_tensor, tensor)
    m.compile(optimizer='sgd', loss='mse')
    pred = m.predict(X_test[1:2])
    reshaped = pred.reshape(26, 26, 8)
    tiled = tile_raster_images(X=reshaped.T, img_shape=(26, 26), tile_shape=(2, 4), tile_spacing=(1, 1))
    image = Image.fromarray(tiled)
    image.save('residual_images/output_at_layer_%d.png' % i)
