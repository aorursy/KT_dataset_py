!pip install -q efficientnet
# loading packages

import pandas as pd
import numpy as np

#

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#

import seaborn as sns
import plotly.express as px

#

import os
import random
import re
import math
import time

from tqdm import tqdm
from tqdm.keras import TqdmCallback


from pandas_summary import DataFrameSummary

import warnings


warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
# Setting color palette.
orange_black = [
    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
]

# Setting plot styling.
plt.style.use('ggplot')
# Importing packages

import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from kaggle_datasets import KaggleDatasets

tf.random.set_seed(seed_val)
# Loading image storage buckets
IMG_READ_SIZE = 512
GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-%ix%i' % (IMG_READ_SIZE, IMG_READ_SIZE))
GCS_PATH2 = KaggleDatasets().get_gcs_path('malignant-v2-%ix%i' % (IMG_READ_SIZE, IMG_READ_SIZE))

multiplier = 20

filenames_train = tf.io.gfile.glob([os.path.join(GCS_PATH2, "train%.2i*.tfrec" % i) for i in range(16, 60)])
filenames_train += tf.io.gfile.glob([os.path.join(GCS_PATH, "train%.2i*.tfrec" % i) for i in range(0, 12)])
for i in range(multiplier):
    filenames_train += tf.io.gfile.glob([os.path.join(GCS_PATH2, "train%.2i*.tfrec" % i) for i in range(0, 12)])
filenames_valid = tf.io.gfile.glob([os.path.join(GCS_PATH, "train%.2i*.tfrec" % i) for i in range(12, 15)])

# filenames_train = tf.io.gfile.glob(GCS_PATH2 + '/train*.tfrec')
# np.random.shuffle(filenames_train)
filenames_test = np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec'))
# Setting TPU as main device for training, if you get warnings while working with tpu's ignore them.

DEVICE = 'TPU'
if DEVICE == 'TPU':
    print('connecting to TPU...')
    try:        
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print('Could not connect to TPU')
        tpu = None

    if tpu:
        try:
            print('Initializing  TPU...')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print('TPU initialized')
        except _:
            print('Failed to initialize TPU!')
    else:
        DEVICE = 'GPU'

if DEVICE != 'TPU':
    print('Using default strategy for CPU and single GPU')
    strategy = tf.distribute.get_strategy()

if DEVICE == 'GPU':
    print('Num GPUs Available: ',
          len(tf.config.experimental.list_physical_devices('GPU')))

print('REPLICAS: ', strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
cfg = dict(
           batch_size=32,
           img_size=IMG_READ_SIZE,
    
           lr_start=0.000005,
           lr_max=0.00000125,
           lr_min=0.000001,
           lr_rampup=5,
           lr_sustain=1,
           lr_decay=0.8,
           epochs=12,
    
           transform_prob=1.0,
           rot=180.0,
           shr=2.0,
           hzoom=8.0,
           wzoom=8.0,
           hshift=8.0,
           wshift=8.0,
    
           optimizer='adam',
           label_smooth_fac=0.05,
           tta_steps=20
            
        )
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift,
            width_shift):
    
    ''' Settings for image preparations '''

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0),
        [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0),
        [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([
            one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero,
            zero, one
        ],
                  axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(
        tf.concat(
            [one, zero, height_shift, zero, one, width_shift, zero, zero, one],
            axis=0), [3, 3])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix, shift_matrix))


def transform(image, cfg):
    
    ''' This function takes input images of [: , :, 3] sizes and returns them as randomly rotated, sheared, shifted and zoomed. '''

    DIM = cfg['img_size']
    XDIM = DIM % 2  # fix for size 331

    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32')
    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0, ], DIM // 2 - 1 + idx2[1, ]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])

def prepare_image(img, cfg=None, augment=True):
    
    ''' This function loads the image, resizes it, casts a tensor to a new type float32 in our case, transforms it using the function just above, then applies the augmentations.'''
    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['img_size'], cfg['img_size']],
                          antialias=True)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        if cfg['transform_prob'] > tf.random.uniform([1], minval=0, maxval=1):
            img = transform(img, cfg)

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    return img
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
#         'patient_id': tf.io.FixedLenFeature([], tf.int64),
#         'sex': tf.io.FixedLenFeature([], tf.int64),
#         'age_approx': tf.io.FixedLenFeature([], tf.int64),
#         'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
#         'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64),
#         'width': tf.io.FixedLenFeature([], tf.int64),
#         'height': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    return example['image'], example['target']


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    return example['image'], example['image_name']

def count_data_items(filenames):
    n = [
        int(re.compile(r'-([0-9]*)\.').search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)
def getTrainDataset(files, cfg, augment=True, shuffle=True):
    
    ''' This function reads the tfrecord train images, shuffles them, apply augmentations to them and prepares the data for training. '''
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    if shuffle:
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.map(lambda img, label:
                (prepare_image(img, augment=augment, cfg=cfg), label),
                num_parallel_calls=AUTO)
    ds = ds.batch(cfg['batch_size'] * strategy.num_replicas_in_sync)
    ds = ds.prefetch(AUTO)
    return ds

def getTestDataset(files, cfg, augment=False, repeat=False):
    
    ''' This function reads the tfrecord test images and prepares the data for predicting. '''
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()
    if repeat:
        ds = ds.repeat()
    ds = ds.map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    ds = ds.map(lambda img, idnum:
                (prepare_image(img, augment=augment, cfg=cfg), idnum),
                num_parallel_calls=AUTO)
    ds = ds.batch(cfg['batch_size'] * strategy.num_replicas_in_sync)
    ds = ds.prefetch(AUTO)
    return ds

# def get_model_b3():
    
#     ''' This function gets the layers inclunding efficientnet ones. '''
    
#     model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),
#                                  name='img_input')

#     dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

#     x = efn.EfficientNetB3(include_top=False,
#                            weights='noisy-student',
#                            input_shape=(cfg['img_size'], cfg['img_size'], 3),
#                            pooling='avg')(dummy)
#     x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
#     model = tf.keras.Model(model_input, x)
#     model.summary()
#     return model

# def get_model_b4():
    
#     ''' This function gets the layers inclunding efficientnet ones. '''
    
#     model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),
#                                  name='img_input')

#     dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

#     x = efn.EfficientNetB4(include_top=False,
#                            weights='noisy-student',
#                            input_shape=(cfg['img_size'], cfg['img_size'], 3),
#                            pooling='avg')(dummy)
#     x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
#     model = tf.keras.Model(model_input, x)
#     model.summary()
#     return model

# def get_model_b5():
    
#     ''' This function gets the layers inclunding efficientnet ones. '''
    
#     model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),
#                                  name='img_input')

#     dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

#     x = efn.EfficientNetB5(include_top=False,
#                            weights='noisy-student',
#                            input_shape=(cfg['img_size'], cfg['img_size'], 3),
#                            pooling='avg')(dummy)
#     x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
#     model = tf.keras.Model(model_input, x)
#     model.summary()
#     return model

# def get_model_b6():
    
#     ''' This function gets the layers inclunding efficientnet ones. '''
    
#     model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),
#                                  name='img_input')

#     dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

#     x = efn.EfficientNetB6(include_top=False,
#                            weights='noisy-student',
#                            input_shape=(cfg['img_size'], cfg['img_size'], 3),
#                            pooling='avg')(dummy)
#     x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
#     model = tf.keras.Model(model_input, x)
#     model.summary()
#     return model
def compileNewModel(cfg, model):
    
    ''' Configuring the model with losses and metrics. '''    

    with strategy.scope():
        model.compile(optimizer=cfg['optimizer'],
                      loss=[
                          tf.keras.losses.BinaryCrossentropy(
                              label_smoothing=cfg['label_smooth_fac'])
                      ],
                      metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def getLearnRateCallback(cfg):
    
    ''' Using callbacks for learning rate adjustments. '''
    
    lr_start = cfg['lr_start']
    lr_max = cfg['lr_max'] * strategy.num_replicas_in_sync * cfg['batch_size']
    lr_min = cfg['lr_min']
    lr_rampup = cfg['lr_rampup']
    lr_sustain = cfg['lr_sustain']
    lr_decay = cfg['lr_decay']

    def lrfn(epoch):
        if epoch < lr_rampup:
            lr = (lr_max - lr_start) / lr_rampup * epoch + lr_start
        elif epoch < lr_rampup + lr_sustain:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_rampup -
                                                lr_sustain) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

def learnModel(model, ds_train, stepsTrain, cfg, ds_val=None, stepsVal=0, fold=0):
    
    ''' Fitting things together for training '''
    cpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    'B3ensemble.h5', monitor='val_auc', verbose=1, save_best_only=True,
                     mode='max', save_freq='epoch')
    
    callbacks = [cpoint_callback, getLearnRateCallback(cfg)]

    history = model.fit(ds_train,
                        validation_data=ds_val,
                        verbose=1,
                        steps_per_epoch=stepsTrain,
                        validation_steps=stepsVal,
                        epochs=cfg['epochs'],
                        callbacks=callbacks)

    return history
ds_train = getTrainDataset(
    filenames_train, cfg).map(lambda img, label: ((img, img, img, img, img), label))#(label, label, label, label)))
stepsTrain = count_data_items(filenames_train) / \
    (cfg['batch_size'] * strategy.num_replicas_in_sync)
ds_val = getTrainDataset(
    filenames_valid, cfg).map(lambda img, label: ((img, img, img, img, img), label))#(label, label, label, label)))
stepsVal = count_data_items(filenames_valid) / \
    (cfg['batch_size'] * strategy.num_replicas_in_sync)
with strategy.scope():
    model1 = tf.keras.models.load_model('../input/b3-models/B3-fold-0.h5')
    model2 = tf.keras.models.load_model('../input/b3-models/B3-fold-1.h5')
    model3 = tf.keras.models.load_model('../input/b3-models/B3-fold-2.h5')
    model4 = tf.keras.models.load_model('../input/b3-models/B3-fold-3.h5')
    model5 = tf.keras.models.load_model('../input/b3-models/B3-fold-4.h5')
def ensemble_model(models):
    for i, model in enumerate(models):
        for j, layer in enumerate(model.layers):
            layer.trainable = False
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name 
            if j>2:
                layer.trainable = True

    ensemble_visible = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]
    merge = tf.keras.layers.concatenate(ensemble_outputs)
    merge = tf.keras.layers.Dense(16, activation='relu')(merge)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(merge)
    model = tf.keras.models.Model(inputs=ensemble_visible, outputs=output)
    return model

models = [model1, model2, model3, model4, model5]
with strategy.scope():
    model = ensemble_model(models)
model = compileNewModel(cfg, model)
model.summary()

ds_train = getTrainDataset(
    filenames_train, cfg).map(lambda img, label: ((img, img, img, img, img), label))#(label, label, label, label)))
stepsTrain = count_data_items(filenames_train) / \
    (cfg['batch_size'] * strategy.num_replicas_in_sync)

learnModel(model, ds_train, stepsTrain, cfg, ds_val, stepsVal)
cfg['batch_size'] = 95
steps = count_data_items(filenames_test) / \
    (cfg['batch_size'] * strategy.num_replicas_in_sync)
z = np.zeros((cfg['batch_size'] * strategy.num_replicas_in_sync))
ds_testAug = getTestDataset(
    filenames_test, cfg, augment=True,
    repeat=True).map(lambda img, label: ((img, img, img, img, img), z))#(z, z, z, z)))

def find_probabilty(model, ds_testAug, steps, cfg, filenames_test, csv_name='sub.csv'):
    probs = model.predict(ds_testAug, verbose=1, steps=steps * cfg['tta_steps'])
    probs = np.stack(probs)
    probs = probs[:, :count_data_items(filenames_test) * cfg['tta_steps']]
    probs = np.stack(np.split(probs, cfg['tta_steps']), axis=1)
    probs = np.mean(probs, axis=1)

    test = pd.read_csv('../input/test-csv/test.csv')
    y_test_sorted = np.zeros((1, probs.shape[1]))
    test = test.reset_index()
    test = test.set_index('image_name')


    ds_test = getTestDataset(filenames_test, cfg)

    image_names = np.array([img_name.numpy().decode("utf-8") 
                            for img, img_name in iter(ds_test.unbatch())])
    
    submission = pd.DataFrame(dict(
        image_name = image_names,
        target     = probs[:,0]))
    
    submission = submission.sort_values('image_name') 
    submission.to_csv(csv_name, index=False)
    return(submission)

        
csv_name = 'B3-512x512-ensemble.csv' 
with strategy.scope():
    model = tf.keras.models.load_model('./B3ensemble.h5')
submission = find_probabilty(model, ds_testAug, steps, cfg, filenames_test, csv_name)


sample.head()