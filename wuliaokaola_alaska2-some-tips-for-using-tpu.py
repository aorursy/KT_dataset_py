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

from sklearn.model_selection import KFold, StratifiedKFold



import tensorflow as tf

from tensorflow.keras import backend as K, layers, losses, optimizers, initializers, constraints, callbacks, models, utils, applications
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
FOLD_ID = 0

TRAIN_BATCH_SIZE = 128

VALID_BATCH_SIZE = 128

AUTO = tf.data.experimental.AUTOTUNE

option_no_order = tf.data.Options()

option_no_order.experimental_deterministic = False
GCS_DS_PATH_Cover = KaggleDatasets().get_gcs_path('alaska2-ds-0512-cover')

GCS_DS_PATH_JMiPOD = KaggleDatasets().get_gcs_path('alaska2-ds-0512-jmipod')

GCS_DS_PATH_JUNIWARD = KaggleDatasets().get_gcs_path('alaska2-ds-0512-juniward')

GCS_DS_PATH_UERD = KaggleDatasets().get_gcs_path('alaska2-ds-0512-uerd')
# The quantization table for quality 95.

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



# The quantization table for quality 90.

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



# The quantization table for quality 75.

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



# The image size is 512x512, so tile quantization table into 64x64



q95 = tf.cast(tf.tile(q95, (64, 64, 1)), dtype=tf.float32)

q90 = tf.cast(tf.tile(q90, (64, 64, 1)), dtype=tf.float32)

q75 = tf.cast(tf.tile(q75, (64, 64, 1)), dtype=tf.float32)



[col, row] = np.meshgrid(range(8), range(8))

T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))

T[0, :] = T[0, :] / np.sqrt(2)

broadcast_dims = (64, 8, 64, 8)

t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)

t = tf.constant(t, dtype=tf.float32)



a = tf.transpose(t, (0, 2, 3, 1))

c = tf.transpose(t, (0, 2, 1, 3))



# Conver DCT into YCbCr

def dct_2_ycc(dct):

    dct = tf.reshape(dct, (64, 8, 64, 8))

    dct = tf.transpose(dct, (0, 2, 1, 3))

    ycc = tf.matmul(tf.matmul(a, dct), c)

    ycc = tf.transpose(ycc, (0, 2, 1, 3))

    ycc = tf.reshape(ycc, (512, 512))

    return ycc
def grid_shuffle(image):

    b = 2 # border size: 8x2=16

    s = 5 # grid size: 8x5=40

    

    m = tf.ones([512 - b * 2 * 8, 512 - b * 2 * 8, 3], dtype=tf.float32)

    m = tf.image.pad_to_bounding_box(m, b * 8, b * 8, 512, 512)

    t = []

    for i in range(b, 64-b, s):

        for j in range(b, 64-b, s):

            t.append([image[i*8:i*8+8*s, j*8:j*8+8*s, :]])

    rn.shuffle(t)

    t = tf.concat(t, axis=0)

    t = tf.reshape(t, ((64-2*b)//s, (64-2*b)//s, 8*s, 8*s, 3))

    t = tf.transpose(t, (4, 0, 2, 1, 3))

    t = tf.reshape(t, (3, 512 - b * 2 * 8, 512 - b * 2 * 8))

    

    t = tf.transpose(t, (1, 2, 0))

    t = tf.image.pad_to_bounding_box(t, b * 8, b * 8, 512, 512)

    image = image * (1 - m) + t * m

    

    return image
def read_tfrecord(data):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "dct1": tf.io.FixedLenFeature([], tf.string), 

        "dct2": tf.io.FixedLenFeature([], tf.string), 

        "q": tf.io.FixedLenFeature([], tf.int64),  

        "mask": tf.io.FixedLenFeature([], tf.string), 

        "label": tf.io.FixedLenFeature([], tf.int64),  

    }



    # decode the TFRecord

    tf_record = tf.io.parse_single_example(data, features)

    

    q = tf.cast(tf_record['q'], tf.float32)

    label = tf.cast(tf_record['label'], tf.float32)

    

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

    

    mask = tf.image.decode_png(tf_record['mask'], channels=1)

    return (ycc, q), (label, mask)



def data_augment(i, o):

    image, q = i

    label, mask = o



    c0 = tf.constant(0)



    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    k = tf.random.uniform([], minval=0, maxval=2, dtype=tf.dtypes.int32)

    image = tf.cond(tf.equal(k, c0), lambda:(image), lambda:(grid_shuffle(image)))

    

    k = tf.random.uniform([], minval=0, maxval=2, dtype=tf.dtypes.int32)

    image = tf.cond(tf.equal(k, c0), lambda:(image), lambda:(-image))

    

    image = tf.reshape(image, [512, 512, 3])

    return (image, q), (label, mask)



def mask_2_payload(i, o):

    image, q = i

    label, mask = o

    

    mask = tf.cast(mask, tf.float32)

    f0 = tf.constant(0, dtype=tf.float32)

    label = tf.cond(tf.equal(K.sum(mask), f0), lambda: f0, lambda: label)

    payload = K.mean(mask)

    return (image, q), (label, payload)
def get_train_dataset(ids):

    filenames = []

    for i in ids:

        filenames.append(GCS_DS_PATH_Cover + "/Cover_{:03d}.tfrec".format(i))

        filenames.append(GCS_DS_PATH_JMiPOD + "/JMiPOD_{:03d}.tfrec".format(i))

        filenames.append(GCS_DS_PATH_JUNIWARD + "/JUNIWARD_{:03d}.tfrec".format(i))

        filenames.append(GCS_DS_PATH_UERD + "/UERD_{:03d}.tfrec".format(i))

    rn.shuffle(filenames)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(option_no_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.map(mask_2_payload, num_parallel_calls=AUTO)

    dataset = dataset.prefetch(AUTO)

    dataset = dataset.shuffle(1024*1)

    dataset = dataset.repeat()

    dataset = dataset.batch(TRAIN_BATCH_SIZE)

    return dataset



def get_valid_dataset(ids):

    filenames = []

    for i in ids:

        filenames.append(GCS_DS_PATH_Cover + "/Cover_{:03d}.tfrec".format(i))

        filenames.append(GCS_DS_PATH_JMiPOD + "/JMiPOD_{:03d}.tfrec".format(i))

        filenames.append(GCS_DS_PATH_JUNIWARD + "/JUNIWARD_{:03d}.tfrec".format(i))

        filenames.append(GCS_DS_PATH_UERD + "/UERD_{:03d}.tfrec".format(i))

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.map(mask_2_payload, num_parallel_calls=AUTO)

    dataset = dataset.prefetch(AUTO)

    dataset = dataset.batch(VALID_BATCH_SIZE)

    return dataset
fid = 0

n = len(glob.glob('../input/alaska2-ds-0512-cover/*.tfrec'))

ids = np.arange(n)

kf = KFold(n_splits=5)

for train_index, valid_index in kf.split(ids):

    train_dataset = get_train_dataset(train_index)    

    valid_dataset = get_valid_dataset(valid_index)

    if fid == FOLD_ID:

        break

    fid += 1
for (images, qs), (labels, payloads) in train_dataset:

    for i in range(10):

        plt.figure(figsize=(10, 10))

        plt.imshow(images[i][:, :, 0])

        plt.show()

    break
def create_model():

    q_input = layers.Input(shape=(1,))

    x_input = layers.Input(shape=(512, 512, 3))

    x = x_input

    q = tf.reshape(q_input, (-1,1,1,1))

    q = tf.tile(q, (1,512,512,1))

    x = layers.Concatenate()([x, q])

    

    mt = efn.EfficientNetB0(

                input_tensor=x,

                weights=None,

                include_top=False

            )

    x = mt.layers[-1].output

    

    gp = layers.GlobalAveragePooling2D()(x)

    l = layers.Dense(128, activation='relu')(gp)

    l = layers.Dense(4, activation='softmax', name='label')(l)

    

    p = layers.Dense(128, activation='relu')(gp)

    p = layers.Dense(1, activation='relu', name='payload')(p)

    

    model = models.Model(inputs=[x_input, q_input], outputs=[l, p])

    return model
with strategy.scope():

    lr_schedule = optimizers.schedules.ExponentialDecay(

        initial_learning_rate=0.001,

        decay_steps=int(75000*4*0.8),

        decay_rate=0.98)

    optimizer = optimizers.Adamax(learning_rate=lr_schedule)

    model = create_model()

    model.compile(

            optimizer=optimizer,

            loss = {'label': 'sparse_categorical_crossentropy', 'payload':'mae'},

            loss_weights = {'label': 1, 'payload':1},

            metrics={'label': 'sparse_categorical_accuracy', 'payload':'accuracy'}

        )
class ValidCB(callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        print('Calculating metric...')

        



model.fit(train_dataset, 

        steps_per_epoch=int(75000*4*0.8/TRAIN_BATCH_SIZE), 

        verbose=1,

        epochs=1,

        initial_epoch=0,

#         validation_data=valid_dataset,

        callbacks=[ValidCB()]

        )