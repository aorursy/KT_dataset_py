!pip install -q git+https://github.com/qubvel/efficientnet
import os

import json

import cv2

import glob

import datetime

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn import metrics



import efficientnet.tfkeras as efn

import numpy as np

import random as rn

import pandas as pd

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

from tensorflow.keras import backend as K, layers, losses, optimizers, initializers, constraints, callbacks, models, utils, applications
# GCS_DS_PATH_Test = KaggleDatasets().get_gcs_path('alaska2-ds-0512-test')

GCS_DS_PATH_Test = '../input/alaska2-ds-0512-test'
q95 = np.array([[[ 2.,  2.,  2.],

        [ 1.,  2.,  2.],

        [ 1.,  2.,  2.],

        [ 2.,  5.,  5.],

        [ 2., 10., 10.],

        [ 4., 10., 10.],

        [ 5., 10., 10.],

        [ 6., 10., 10.]],



       [[ 1.,  2.,  2.],

        [ 1.,  2.,  2.],

        [ 1.,  3.,  3.],

        [ 2.,  7.,  7.],

        [ 3., 10., 10.],

        [ 6., 10., 10.],

        [ 6., 10., 10.],

        [ 6., 10., 10.]],



       [[ 1.,  2.,  2.],

        [ 1.,  3.,  3.],

        [ 2.,  6.,  6.],

        [ 2., 10., 10.],

        [ 4., 10., 10.],

        [ 6., 10., 10.],

        [ 7., 10., 10.],

        [ 6., 10., 10.]],



       [[ 1.,  5.,  5.],

        [ 2.,  7.,  7.],

        [ 2., 10., 10.],

        [ 3., 10., 10.],

        [ 5., 10., 10.],

        [ 9., 10., 10.],

        [ 8., 10., 10.],

        [ 6., 10., 10.]],



       [[ 2., 10., 10.],

        [ 2., 10., 10.],

        [ 4., 10., 10.],

        [ 6., 10., 10.],

        [ 7., 10., 10.],

        [11., 10., 10.],

        [10., 10., 10.],

        [ 8., 10., 10.]],



       [[ 2., 10., 10.],

        [ 4., 10., 10.],

        [ 6., 10., 10.],

        [ 6., 10., 10.],

        [ 8., 10., 10.],

        [10., 10., 10.],

        [11., 10., 10.],

        [ 9., 10., 10.]],



       [[ 5., 10., 10.],

        [ 6., 10., 10.],

        [ 8., 10., 10.],

        [ 9., 10., 10.],

        [10., 10., 10.],

        [12., 10., 10.],

        [12., 10., 10.],

        [10., 10., 10.]],



       [[ 7., 10., 10.],

        [ 9., 10., 10.],

        [10., 10., 10.],

        [10., 10., 10.],

        [11., 10., 10.],

        [10., 10., 10.],

        [10., 10., 10.],

        [10., 10., 10.]]])



q90 = np.array([[[ 3.,  3.,  3.],

        [ 2.,  4.,  4.],

        [ 2.,  5.,  5.],

        [ 3.,  9.,  9.],

        [ 5., 20., 20.],

        [ 8., 20., 20.],

        [10., 20., 20.],

        [12., 20., 20.]],



       [[ 2.,  4.,  4.],

        [ 2.,  4.,  4.],

        [ 3.,  5.,  5.],

        [ 4., 13., 13.],

        [ 5., 20., 20.],

        [12., 20., 20.],

        [12., 20., 20.],

        [11., 20., 20.]],



       [[ 3.,  5.,  5.],

        [ 3.,  5.,  5.],

        [ 3., 11., 11.],

        [ 5., 20., 20.],

        [ 8., 20., 20.],

        [11., 20., 20.],

        [14., 20., 20.],

        [11., 20., 20.]],



       [[ 3.,  9.,  9.],

        [ 3., 13., 13.],

        [ 4., 20., 20.],

        [ 6., 20., 20.],

        [10., 20., 20.],

        [17., 20., 20.],

        [16., 20., 20.],

        [12., 20., 20.]],



       [[ 4., 20., 20.],

        [ 4., 20., 20.],

        [ 7., 20., 20.],

        [11., 20., 20.],

        [14., 20., 20.],

        [22., 20., 20.],

        [21., 20., 20.],

        [15., 20., 20.]],



       [[ 5., 20., 20.],

        [ 7., 20., 20.],

        [11., 20., 20.],

        [13., 20., 20.],

        [16., 20., 20.],

        [21., 20., 20.],

        [23., 20., 20.],

        [18., 20., 20.]],



       [[10., 20., 20.],

        [13., 20., 20.],

        [16., 20., 20.],

        [17., 20., 20.],

        [21., 20., 20.],

        [24., 20., 20.],

        [24., 20., 20.],

        [20., 20., 20.]],



       [[14., 20., 20.],

        [18., 20., 20.],

        [19., 20., 20.],

        [20., 20., 20.],

        [22., 20., 20.],

        [20., 20., 20.],

        [21., 20., 20.],

        [20., 20., 20.]]])



q75 = np.array([[[ 8.,  9.,  9.],

        [ 6.,  9.,  9.],

        [ 5., 12., 12.],

        [ 8., 24., 24.],

        [12., 50., 50.],

        [20., 50., 50.],

        [26., 50., 50.],

        [31., 50., 50.]],



       [[ 6.,  9.,  9.],

        [ 6., 11., 11.],

        [ 7., 13., 13.],

        [10., 33., 33.],

        [13., 50., 50.],

        [29., 50., 50.],

        [30., 50., 50.],

        [28., 50., 50.]],



       [[ 7., 12., 12.],

        [ 7., 13., 13.],

        [ 8., 28., 28.],

        [12., 50., 50.],

        [20., 50., 50.],

        [29., 50., 50.],

        [35., 50., 50.],

        [28., 50., 50.]],



       [[ 7., 24., 24.],

        [ 9., 33., 33.],

        [11., 50., 50.],

        [15., 50., 50.],

        [26., 50., 50.],

        [44., 50., 50.],

        [40., 50., 50.],

        [31., 50., 50.]],



       [[ 9., 50., 50.],

        [11., 50., 50.],

        [19., 50., 50.],

        [28., 50., 50.],

        [34., 50., 50.],

        [55., 50., 50.],

        [52., 50., 50.],

        [39., 50., 50.]],



       [[12., 50., 50.],

        [18., 50., 50.],

        [28., 50., 50.],

        [32., 50., 50.],

        [41., 50., 50.],

        [52., 50., 50.],

        [57., 50., 50.],

        [46., 50., 50.]],



       [[25., 50., 50.],

        [32., 50., 50.],

        [39., 50., 50.],

        [44., 50., 50.],

        [52., 50., 50.],

        [61., 50., 50.],

        [60., 50., 50.],

        [51., 50., 50.]],



       [[36., 50., 50.],

        [46., 50., 50.],

        [48., 50., 50.],

        [49., 50., 50.],

        [56., 50., 50.],

        [50., 50., 50.],

        [52., 50., 50.],

        [50., 50., 50.]]])

q95 = tf.cast(tf.tile(q95, (64,64,1)), dtype=tf.float32)

q90 = tf.cast(tf.tile(q90, (64,64,1)), dtype=tf.float32)

q75 = tf.cast(tf.tile(q75, (64,64,1)), dtype=tf.float32)



[col, row] = np.meshgrid(range(8), range(8))

T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))

T[0, :] = T[0, :] / np.sqrt(2)

broadcast_dims = (64, 8, 64, 8)

t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)

t = tf.constant(t, dtype=tf.float32)



a = tf.transpose(t, (0, 2, 3, 1))

c = tf.transpose(t, (0, 2, 1, 3))
def dct_2_ycc(dct):

    dct = tf.reshape(dct, (64, 8, 64, 8))

    dct = tf.transpose(dct, (0,2,1,3))

    ycc = tf.matmul(tf.matmul(a, dct), c)

    ycc = tf.transpose(ycc, (0, 2, 1, 3))

    ycc = tf.reshape(ycc, (512, 512))

    return ycc



def read_test_tfrecord(data):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "dct1": tf.io.FixedLenFeature([], tf.string), 

        "dct2": tf.io.FixedLenFeature([], tf.string), 

        "q": tf.io.FixedLenFeature([], tf.int64), 

    }



    # decode the TFRecord

    tf_record = tf.io.parse_single_example(data, features)

    

    q = tf.cast(tf_record['q'], tf.float32)

    

    dct1 = tf.image.decode_png(tf_record['dct1'])

    dct2 = tf.image.decode_png(tf_record['dct2'])

    dct = tf.cast(dct1, tf.int32) * 256 + tf.cast(dct2, tf.int32)

    dct = tf.cast(dct, tf.int16)

    

    dct = tf.cast(dct, tf.float32)

    dct = tf.case([(tf.math.equal(q, tf.constant(0, dtype=tf.float32)), lambda: dct * q95), 

                (tf.math.equal(q, tf.constant(1, dtype=tf.float32)), lambda: dct * q90),

                (tf.math.equal(q, tf.constant(2, dtype=tf.float32)), lambda: dct * q75),

                ])

    

    ycc = tf.stack([dct_2_ycc(dct[:,:,0]), dct_2_ycc(dct[:,:,1]), dct_2_ycc(dct[:,:,2])], axis=2) / 255.

    

    return (ycc, q), (q, q)



def tta_flip_left_right(i, o):

    image, q = i

    image = tf.image.flip_left_right(image)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o



def tta_flip_up_down(i, o):

    image, q = i

    image = tf.image.flip_up_down(image)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o



def tta_rot90(i, o):

    image, q = i

    image = tf.image.rot90(image, k=1)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o



def tta_rot180(i, o):

    image, q = i

    image = tf.image.rot90(image, k=2)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o



def tta_rot270(i, o):

    image, q = i

    image = tf.image.rot90(image, k=3)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o



def tta_rot90_flip_left_right(i, o):

    image, q = i

    image = tf.image.rot90(image, k=1)

    image = tf.image.flip_left_right(image)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o



def tta_rot90_flip_up_down(i, o):

    image, q = i

    image = tf.image.rot90(image, k=1)

    image = tf.image.flip_up_down(image)

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), o
def get_test_dataset(tta=None):

    filenames = sorted(tf.io.gfile.glob(GCS_DS_PATH_Test + '/*.tfrec'))

    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(read_test_tfrecord)

    if tta is not None:

        dataset = dataset.map(tta)

    dataset = dataset.batch(32)

    return dataset
def create_model():

    q_input = layers.Input(shape=(1,))

    x_input = layers.Input(shape=(512, 512, 3))

    x = x_input

    q = tf.reshape(q_input, (-1,1,1,1))

    q = tf.tile(q, (1,512,512,1))

    x = layers.Concatenate()([x, q])

    x = layers.BatchNormalization(fused=True)(x)

    

    mt = efn.EfficientNetB6(

                input_tensor=x,

                weights=None,

                include_top=False

            )

    x = mt.layers[-1].output

    

    gp = layers.GlobalAveragePooling2D()(x)

    l = layers.Dense(128, activation='relu', name='ds_l_1')(gp)

    l = layers.Dense(4, activation='softmax', name='label')(l)

    

    p = layers.Dense(128, activation='relu', name='ds_p_1')(gp)

    p = layers.Dense(1, activation='relu', name='payload')(p)

    

    model = models.Model(inputs=[x_input, q_input], outputs=[l, p])

    return model



model = create_model()

model.load_weights('../input/alaska2-ef6-0929/ycc_ef6_f0_l2_ep038_0.54500.h5')
ids = ['{:04d}.jpg'.format(i+1) for i in range(5000)]

sub = pd.DataFrame(columns=['Id', 'Label'])

sub['Id'] = ids



psa = []

psa.append(1 -  model.predict(get_test_dataset(), verbose=1)[0][:, 0])

psa.append(1 -  model.predict(get_test_dataset(tta_flip_left_right), verbose=1)[0][:, 0])

psa.append(1 -  model.predict(get_test_dataset(tta_flip_up_down), verbose=1)[0][:, 0])

psa.append(1 -  model.predict(get_test_dataset(tta_rot180), verbose=1)[0][:, 0])



sub['Label'] = np.mean(psa, axis=0)

sub.to_csv('submission_tta.csv', index=None)