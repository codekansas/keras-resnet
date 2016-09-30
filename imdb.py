"""Train a Bidirectional LSTM on the IMDB sentiment classification task.

Applies residual function to the output of the LSTM, followed by max pooling.
Similar speed and accuracy to the version without residual / max pooling.

See: https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py
"""
from __future__ import print_function
import numpy as np
np.random.seed(42)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, Lambda
from keras.datasets import imdb
import keras.backend as K  # Needed for max pooling operation
from resnet import Residual

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

input = Input(shape=(maxlen,))
embedded = Embedding(max_features, 128)(input)

def get_lstm_model():
    input = Input(shape=(maxlen, 128))
    output = Bidirectional(LSTM(64, return_sequences=True))(input)
    return Model(input, output)

resnet = Residual(get_lstm_model())(embedded)
maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False),
                 output_shape=lambda x: (x[0], x[2]))(resnet)
dropout = Dropout(0.5)(maxpool)
output = Dense(1, activation='sigmoid')(dropout)
model = Model(input=input, output=output)

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=4,
          validation_data=[X_test, y_test])
