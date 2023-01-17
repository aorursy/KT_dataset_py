import numpy as np

import pandas as pd

import os

import psutil

import math



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import time

import datetime



import gc

import seaborn as sns

import tensorflow as tf



import warnings

warnings.filterwarnings("ignore")
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
X_mean = train['acoustic_data'].mean()

X_std = train['acoustic_data'].std()

y_mean = train['time_to_failure'].mean()

y_std = train['time_to_failure'].std()
cut_time = []

for i in range(0,len(train)-10000,10000):

    if train['time_to_failure'][i] < train['time_to_failure'][i + 10000]:

        cut_time.append(i)

accurate_cut_time = [0]

for t in cut_time:

    for i in range(t,t+10000):

        if train['time_to_failure'][i] < train['time_to_failure'][i + 1]:

            accurate_cut_time.append(i + 1)

            break

accurate_cut_time.append(len(train))
df_all = []

for i in range(len(accurate_cut_time)-1):

    df_all.append(train.iloc[accurate_cut_time[i]:accurate_cut_time[i] + 150000 * ((accurate_cut_time[i+1] - accurate_cut_time[i]) // 150000)])
del train

del cut_time

del accurate_cut_time

gc.collect()
val_df = []

val_target = []

for df in df_all[:3]:

    segments = int(np.floor(df.shape[0] / 150000))

    for segment in tqdm_notebook(range(segments)):

        seg = df.iloc[segment*150000:segment*150000+150000]

        x_raw = seg['acoustic_data']

        val_df.append((x_raw.values - X_mean) / X_std)

        val_target.append(seg['time_to_failure'].values[-1])

val_df = np.array(val_df).reshape((-1, 150000,1))

val_target = np.array(val_target).reshape((-1, 1))
import keras

from keras.models import Model

from keras.layers import *

from keras.optimizers import Adam,SGD

from keras.backend import clear_session

import tensorflow.keras.backend as K

import tensorflow as tf



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
clear_session()

inp = Input(shape=(150000,1))



# First_LSTM_Cell = Bidirectional(CuDNNLSTM(32, return_sequences=True))

# First_Attention = Attention(400)

# x = []

# for i in range(375):

#     temp = First_LSTM_Cell(Lambda(lambda x: keras.backend.slice(x, (0,i*400,0), (-1,400,-1)))(inp))

#     x.append(First_Attention(temp))

# x = Concatenate()(x)

# x = Reshape((375,128))(x)



x = inp

dilation_rates = [2**i for i in range(10)] 

for dilation_rate in dilation_rates:

    tanh_out = Conv1D(filters=16,

            kernel_size=15, 

            padding='same',

            activation = 'tanh',

            dilation_rate=dilation_rate)(x)

    sigm_out = Conv1D(filters=16,

            kernel_size=15, 

            padding='same',

            activation = 'sigmoid',

            dilation_rate=dilation_rate)(x)

    x = Multiply()([tanh_out,sigm_out])

    x = Conv1D(filters = 16,

                   kernel_size = 1,

                   padding='same',

                   activation = 'relu',)(x)

    x = BatchNormalization()(x)

    x = SpatialDropout1D(0.2)(x)



x = Conv1D(filters=16,

        kernel_size=60, 

        strides = 20,

#         activation='relu',

        padding='same')(inp)



dilation_rates = [2**i for i in range(5)] 

for dilation_rate in dilation_rates:

    tanh_out = Conv1D(filters=32,

            kernel_size=15, 

            padding='same',

            activation = 'tanh',

            dilation_rate=dilation_rate)(x)

    sigm_out = Conv1D(filters=32,

            kernel_size=15, 

            padding='same',

            activation = 'sigmoid',

            dilation_rate=dilation_rate)(x)

    x = Multiply()([tanh_out,sigm_out])

    x = Conv1D(filters = 32,

                   kernel_size = 1,

                   padding='same',

                   activation = 'relu',)(x)

    x = BatchNormalization()(x)

    x = SpatialDropout1D(0.2)(x)

    

x = Conv1D(filters=32,

        kernel_size=60, 

        strides = 20,

#         activation='relu',

        padding='same')(x)



x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = Attention(375)(x)

x = Dropout(0.2)(x)

x = Dense(64, activation="relu")(x)

x = Dense(1)(x)



model = Model(inputs = inp, outputs=x)

model.summary()
def generator():

    batch_size = 16

    while True:

        X = []

        Y = []

        for i in range(batch_size):

            while True:

                land = 3 + np.random.randint(len(df_all)-3)

                point = np.random.randint(int((len(df_all[land])-150000)/100))

                seg = df_all[land].iloc[point*100:point*100+150000]

                if seg['time_to_failure'].values[-1] < seg['time_to_failure'].values[0]:

                    break

            x_raw = seg['acoustic_data']

            x = (x_raw.values - X_mean) / X_std

            y = seg['time_to_failure'].values[-1]

#             y = (y - y_mean) / y_std

            X.append(x)

            Y.append(y)

        yield np.array(X).reshape((-1, 150000,1)),np.array(Y).reshape((-1, 1))
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping

def step_decay(epoch):

    x = 0.0001

    if epoch >= 200: x = 0.00001

    if epoch >= 500: x = 0.000001

    return x

lr_decay = LearningRateScheduler(step_decay)
model.compile(Adam(lr=0.0001), loss='mae')

history = model.fit_generator(generator(),

                             steps_per_epoch=400,

                             epochs=1000,

                             verbose=2,

                             callbacks = [ModelCheckpoint("model.hdf5", save_best_only=True,save_weights_only = True, period=1), lr_decay, EarlyStopping(patience = 50)],

                             validation_data = (val_df,val_target)

                             )
model.load_weights("model.hdf5")
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

pred_test = []

for i, seg_id in enumerate(tqdm_notebook(submission.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x_raw = seg['acoustic_data']

    pred_test.append(model.predict(((x_raw.values - X_mean) / X_std).reshape((-1, 150000,1))).reshape((1)))

submission['time_to_failure'] = np.array(pred_test)

print(submission.head())

submission.to_csv('submission.csv')