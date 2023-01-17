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
# Setting file paths for our notebook:

base_path = '/kaggle/input/siim-isic-melanoma-classification'
train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
img_stats_path = '/kaggle/input/melanoma2020imgtabular'

train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from kaggle_datasets import KaggleDatasets

tf.random.set_seed(seed_val)
# Loading image storage buckets

GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-384x384')
filenames_train = np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'))
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
           batch_size=64,
           img_size=384,
    
           lr_start=0.00001,
           lr_max=0.00000125,
           lr_min=0.000001,
           lr_rampup=5,
           lr_sustain=0,
           lr_decay=0.8,
           epochs=15,
    
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
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64)
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

def get_model():
    
    ''' This function gets the layers inclunding efficientnet ones. '''
    
    model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),
                                 name='img_input')

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    x = efn.EfficientNetB3(include_top=False, # USE WHICHEVER YOU WANT HERE
                           weights='noisy-student',
                           input_shape=(cfg['img_size'], cfg['img_size'], 3),
                           pooling='avg')(dummy) 
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(model_input, x)
    model.summary()
    return model
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

def learnModel(model, ds_train, stepsTrain, cfg, ds_val=None, stepsVal=0):
    
    ''' Fitting things together for training '''
    
    callbacks = [getLearnRateCallback(cfg)]

    history = model.fit(ds_train,
                        validation_data=ds_val,
                        verbose=1,
                        steps_per_epoch=stepsTrain,
                        validation_steps=stepsVal,
                        epochs=cfg['epochs'],
                        callbacks=callbacks)

    return history
ds_train = getTrainDataset(
    filenames_train, cfg).map(lambda img, label: (img, label))#(label, label, label, label)))
stepsTrain = count_data_items(filenames_train) / \
    (cfg['batch_size'] * strategy.num_replicas_in_sync)
with strategy.scope():
    model = get_model()

model = compileNewModel(cfg, model)
history = learnModel(model, ds_train, stepsTrain, cfg)
cfg['optimizer']='sgd'
with strategy.scope():
    model1 = get_model()

model1 = compileNewModel(cfg, model1)
history1 = learnModel(model1, ds_train, stepsTrain, cfg)
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


class AdaBound(OptimizerV2):
    """AdaBound optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        final_learning_rate: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.
    # References
        - [Adaptive Gradient Methods with Dynamic Bound of Learning Rate]
          (https://openreview.net/forum?id=Bkg3g2R9FX)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """
    def __init__(self,
                 learning_rate=0.001,
                 final_learning_rate=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 gamma=1e-3,
                 epsilon=None,
                 weight_decay=0.0,
                 amsbound=False,
                 name='AdaBound', **kwargs):
        super(AdaBound, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('learning_rate', learning_rate))
        self._set_hyper('final_learning_rate', kwargs.get('final_learning_rate', final_learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('gamma', gamma)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsbound = amsbound
        self.weight_decay = weight_decay
        self.base_lr = learning_rate

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'vhat')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        vhat = self.get_slot(var, 'vhat')

        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)

        gamma = self._get_hyper('gamma')
        final_lr = self._get_hyper('final_learning_rate')

        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        base_lr_t = tf.convert_to_tensor(self.base_lr)
        t = tf.cast(self.iterations + 1, var_dtype)

        # Applies bounds on actual learning rate
        step_size = lr_t * (tf.math.sqrt(1. - tf.math.pow(beta_2_t, t)) /
                          (1. - tf.math.pow(beta_1_t, t)))

        final_lr = final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma * t))

        # apply weight decay
        if self.weight_decay != 0.:
            grad += self.weight_decay * var

        # Compute moments
        m_t = (beta_1_t * m) + (1. - beta_1_t) * grad
        v_t = (beta_2_t * v) + (1. - beta_2_t) * tf.math.square(grad)

        if self.amsbound:
            vhat_t = tf.math.maximum(vhat, v_t)
            denom = (tf.math.sqrt(vhat_t) + epsilon_t)
        else:
            vhat_t = vhat
            denom = (tf.math.sqrt(v_t) + self.epsilon)

        # Compute the bounds
        step_size_p = step_size * tf.ones_like(denom)
        step_size_p_bound = step_size_p / denom
        bounded_lr_t = m_t * tf.math.minimum(tf.math.maximum(step_size_p_bound,
                                             lower_bound), upper_bound)

        # Setup updates
        m_t = tf.compat.v1.assign(m, m_t)
        vhat_t = tf.compat.v1.assign(vhat, vhat_t)

        with tf.control_dependencies([m_t, v_t, vhat_t]):
            p_t = var - bounded_lr_t
            param_update = tf.compat.v1.assign(var, p_t)

            return tf.group(*[param_update, m_t, v_t, vhat_t])

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("Sparse data is not supported yet")

    def get_config(self):
        config = super(AdaBound, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'final_learning_rate': self._serialize_hyperparameter('final_learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'gamma': self._serialize_hyperparameter('gamma'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'amsbound': self.amsbound,
        })
        return config
cfg['optimizer'] = AdaBoundaBoundaBoost(amsbound=False)
with strategy.scope():
    model2 = get_model()

model2 = compileNewModel(cfg, model2)
history2 = learnModel(model2, ds_train, stepsTrain, cfg)
cfg['optimizer'] = 'adam'
cfg['epochs'] = 25
with strategy.scope():
    model2 = get_model()

model2 = compileNewModel(cfg, model2)
history2 = learnModel(model2, ds_train, stepsTrain, cfg)
cfg['epochs'] = 1
history2 = learnModel(model2, ds_train, stepsTrain, cfg)
auc = history2.history['auc']
loss = history2.history['loss']
epochs_range = range(cfg['epochs'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, auc, label='adam')

plt.legend(loc='lower right')
plt.title('Training and AUC')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='adam')

plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()
auc = history.history['auc']
loss = history.history['loss']
auc1 = history1.history['auc']
loss1 = history1.history['loss']
auc2 = history2.history['auc']
loss2 = history2.history['loss']
epochs_range = range(cfg['epochs'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, auc, label='adam')
plt.plot(epochs_range, auc1, label='sgd')
plt.plot(epochs_range, auc2, label='adabound')

plt.legend(loc='lower right')
plt.title('Training and AUC')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='adam')
plt.plot(epochs_range, loss1, label='sgd')
plt.plot(epochs_range, loss2, label='adabound')

plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()
cfg['batch_size'] = 95
steps = count_data_items(filenames_test) / \
    (cfg['batch_size'] * strategy.num_replicas_in_sync)
z = np.zeros((cfg['batch_size'] * strategy.num_replicas_in_sync))
ds_testAug = getTestDataset(
    filenames_test, cfg, augment=True,
    repeat=True).map(lambda img, label: (img, z))#(z, z, z, z)))
probs = model2.predict(ds_testAug, verbose=1, steps=steps * cfg['tta_steps'])
probs = np.stack(probs)
probs = probs[:, :count_data_items(filenames_test) * cfg['tta_steps']]
probs = np.stack(np.split(probs, cfg['tta_steps']), axis=1)
probs = np.mean(probs, axis=1)

test = pd.read_csv(os.path.join(base_path, 'test.csv'))
y_test_sorted = np.zeros((1, probs.shape[1]))
test = test.reset_index()
test = test.set_index('image_name')


ds_test = getTestDataset(filenames_test, cfg)

image_names = np.array([img_name.numpy().decode("utf-8") 
                        for img, img_name in iter(ds_test.unbatch())])
for i in range(1):
    submission = pd.DataFrame(dict(
        image_name = image_names,
        target     = probs[:,0]))

    submission = submission.sort_values('image_name') 
    submission.to_csv('effnetsB.csv', index=False)