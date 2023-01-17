import copy

import gc

import glob

import os

import time



import cv2

import IPython

import IPython.display

import joblib

import librosa

import librosa.display

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy

from joblib import Parallel, delayed

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tqdm import tqdm



%matplotlib inline
pd.options.display.max_columns = 128

pd.options.display.max_rows = 128

plt.rcParams['figure.figsize'] = (15, 8)
class EasyDict(dict):

    def __init__(self, d=None, **kwargs):

        if d is None:

            d = {}

        if kwargs:

            d.update(**kwargs)

        for k, v in d.items():

            setattr(self, k, v)

        # Class attributes

        for k in self.__class__.__dict__.keys():

            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):

                setattr(self, k, getattr(self, k))



    def __setattr__(self, name, value):

        if isinstance(value, (list, tuple)):

            value = [self.__class__(x)

                     if isinstance(x, dict) else x for x in value]

        elif isinstance(value, dict) and not isinstance(value, self.__class__):

            value = self.__class__(value)

        super(EasyDict, self).__setattr__(name, value)

        super(EasyDict, self).__setitem__(name, value)



    __setitem__ = __setattr__



    def update(self, e=None, **f):

        d = e or dict()

        d.update(f)

        for k in d:

            setattr(self, k, d[k])



    def pop(self, k, d=None):

        delattr(self, k)

        return super(EasyDict, self).pop(k, d)
train_df = pd.read_csv('../input/train_curated.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

print('train: {}'.format(train_df.shape))

print('test: {}'.format(sample_submission.shape))



ROOT = '../input/'

test_root = os.path.join(ROOT, 'test/')

train_root = os.path.join(ROOT, 'train_curated/')





CONFIG = EasyDict()

CONFIG.hop_length = 347 # to make time steps 128

CONFIG.fmin = 20

CONFIG.fmax = 44100 / 2

CONFIG.n_fft = 480



N_SAMPLES = 48

SAMPLE_DIM = 256



TRAINING_CONFIG = {

    'sample_dim': (N_SAMPLES, SAMPLE_DIM),

    'padding_mode': cv2.BORDER_REFLECT,

}



print(CONFIG)

print(TRAINING_CONFIG)



train_df.head()
# Preprocessing functions inspired by:

# https://github.com/xiaozhouwang/tensorflow_speech_recognition_solution/blob/master/data.py

class DataProcessor(object):

    

    def __init__(self, debug=False):

        self.debug = debug

        

        # Placeholders for global statistics

        self.mel_mean = None

        self.mel_std = None

        self.mel_max = None

        self.mfcc_max = None

        

    def createMel(self, filename, params, normalize=False):

        """

        Create Mel Spectrogram sample out of raw wavfile

        """

        y, sr = librosa.load(filename, sr=None)

        mel = librosa.feature.melspectrogram(y, sr, n_mels=N_SAMPLES, **params)

        mel = librosa.power_to_db(mel)

        if normalize:

            if self.mel_mean is not None and self.mel_std is not None:

                mel = (mel - self.mel_mean) / self.mel_std

            else:

                sample_mean = np.mean(mel)

                sample_std = np.std(mel)

                mel = (mel - sample_mean) / sample_std

            if self.mel_max is not None:

                mel = mel / self.mel_max

            else:

                mel = mel / np.max(np.abs(mel))

        return mel

    

    def createMfcc(self, filename, params, normalize=False):

        """

        Create MFCC sample out of raw wavfile

        """

        y, sr = librosa.load(filename, sr=None)

        nonzero_idx = [y > 0]

        y[nonzero_idx] = np.log(y[nonzero_idx])

        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=N_SAMPLES, **params)

        if normalize:

            if self.mfcc_max is not None:

                mfcc = mfcc / self.mfcc_max

            else:

                mfcc = mfcc / np.max(np.abs(mfcc))

        return mfcc

    

    def createLogspec(self, filename, params,

                      normalize=False,

                      window_size=20,

                      step_size=10, eps=1e-10):

        """

        Create log spectrogram,

        based on 

        https://www.kaggle.com/voglinio/keras-2d-model-5-fold-log-specgram-curated-only

        """

        

        y, sr = librosa.load(filename, sr=None)

        nperseg = int(round(window_size * sr / 1e3))

        noverlap = int(round(step_size * sr / 1e3))

        freqs, times, spec = scipy.signal.spectrogram(

            y,

            fs=sr,

            window='hann',

            nperseg=nperseg,

            noverlap=noverlap,

            detrend=False)

        spec = np.log(spec.astype(np.float32) + eps)

        return spec

        

    

    def prepareSample(self, root, row, 

                      preprocFunc, 

                      preprocParams, trainingParams, 

                      test_mode=False, normalize=False, 

                      proc_mode='split'):

        """

        Prepare sample for model training.

        Function takes row of DataFrame, extracts filename and labels and processes them.

        

        If proc_mode is 'split':

        Outputs sets of arrays of constant shape padded to TRAINING_CONFIG shape

        with selected padding mode, also specified in TRAINING_CONFIG.

        This approach prevents loss of information caused by trimming the audio sample,

        instead it splits it into equally-sized parts and pads them.

        To account for creation of multiple samples, number of labels are multiplied to a number

        equal to number of created samples.

        

        If proc_mode is 'resize':

        Resizes the original processed sample to (SAMPLE_DIM, N_SAMPLES) shape.

        """

        

        assert proc_mode in ['split', 'resize'], 'proc_must be one of split or resize'

        

        filename = os.path.join(root, row['fname'])

        if not test_mode:

            labels = row['labels']

            

        sample = preprocFunc(filename, preprocParams, normalize=normalize)

        # print(sample.min(), sample.max())

        

        if proc_mode == 'split':

            sample_split = np.array_split(

                sample, np.ceil(sample.shape[1] / SAMPLE_DIM), axis=1)

            samples_pad = []

            for i in sample_split:

                padding_dim = SAMPLE_DIM - i.shape[1]

                sample_pad = cv2.copyMakeBorder(i, 0, 0, 0, padding_dim, trainingParams['padding_mode'])

                samples_pad.append(sample_pad)

            samples_pad = np.asarray(samples_pad)

            if not test_mode:

                labels = [labels] * len(samples_pad)

                labels = np.asarray(labels)

                return samples_pad, labels

            return samples_pad

        elif proc_mode == 'resize':

            sample_pad = cv2.resize(sample, (SAMPLE_DIM, N_SAMPLES), interpolation=cv2.INTER_NEAREST)

            sample_pad = np.expand_dims(sample_pad, axis=0)

            if not test_mode:

                labels = np.asarray(labels)

                return sample_pad, labels

            return sample_pad

        elif proc_mode == 'raw':

            if not test_model:

                return sample, labels

            return sample





processor = DataProcessor()
DATA_PREFIX = 'proc'

train_filename = 'train_curated_{}.joblib'.format(DATA_PREFIX)

test_filename = 'test_{}.joblib'.format(DATA_PREFIX)





# Train processing/loading:

if os.path.isfile(train_filename):

    print('load processed train:')

    train_dict = joblib.load(train_filename)

    X_train = train_dict['X']

    y_train = train_dict['y']

    print(X_train.shape, y_train.shape)

else:

    print('process train...')

    output = Parallel(n_jobs=-3, verbose=1)(

        delayed(processor.prepareSample)(

            train_root, 

            train_df.iloc[f, :],

            processor.createLogspec,

            CONFIG,

            TRAINING_CONFIG,

            test_mode=False,

            proc_mode='resize',

        ) for f in range(train_df.shape[0]))  # change to number of sample in train data for full processing

    X_train = np.array([x[0] for x in output])

    y_train = np.array([x[1] for x in output])

    y_train = pd.Series(y_train).str.get_dummies(sep=',')

    print(X_train.shape, y_train.shape)

    # Save output for quicker experiments

    train_dict = {

        'X': X_train,

        'y': y_train,

    }

    joblib.dump(train_dict, train_filename)

    



# Test processing/loading:

if os.path.isfile(test_filename):

    print('load processed test:')

    test_dict = joblib.load(test_filename)

    X_test = test_dict['X']

    print(X_test.shape)

else:

    print('process test...')

    X_test = Parallel(n_jobs=-3, verbose=1)(

        delayed(processor.prepareSample)(

            test_root, 

            sample_submission.iloc[f, :],

            processor.createLogspec,

            CONFIG,

            TRAINING_CONFIG,

            test_mode=True,

            proc_mode='resize',

        ) for f in range(sample_submission.shape[0]))  # change to number of sample in test data for full processing

    X_test = np.array(X_test)

    print(X_test.shape)

    test_dict = {

        'X': X_test,

    }

    joblib.dump(test_dict, test_filename)

    

    

# Switch channel axis from 2nd to last

X_train = np.moveaxis(X_train, 1, -1)

X_test = np.moveaxis(X_test, 1, -1)

print(X_train.shape, y_train.shape)
# Distribution of multilabel labels

print('Multilabel class distribution:')

print(y_train.sum(axis=1).value_counts())



# Most of the samples belong to only one class.

# There are some (around 15%), which belong to two classes.

# Occurrence of samples belonging to more than two classes at once

# is quite rare, around 1.5%.
from collections import Counter

from pprint import pprint



df_lab = (y_train.loc[y_train.sum(axis=1) > 1] > 0)

multilabel_combs = []

for i in range(df_lab.shape[0]):

    row_label = df_lab.iloc[i, :][df_lab.iloc[i, :] > 0].index.tolist()

    multilabel_combs.append(row_label)



multilabel_comb_counter = Counter(list(map(lambda x: ' + '.join(x), multilabel_combs)))

pprint(multilabel_comb_counter.most_common(20))

# 20 most common combinations of labels
import numpy as np

import sklearn.metrics





# Based on https://www.kaggle.com/voglinio/keras-2d-model-5-fold-log-specgram-curated-only

# Core calculation of label precisions for one test sample.

def _one_sample_positive_class_precisions(scores, truth):

    """Calculate precisions for each true class for a single sample.



    Args:

      scores: np.array of (num_classes,) giving the individual classifier scores.

      truth: np.array of (num_classes,) bools indicating which classes are true.



    Returns:

      pos_class_indices: np.array of indices of the true classes for this sample.

      pos_class_precisions: np.array of precisions corresponding to each of those

        classes.

    """

    num_classes = scores.shape[0]

    pos_class_indices = np.flatnonzero(truth > 0)

    # Only calculate precisions if there are some true classes.

    if not len(pos_class_indices):

        return pos_class_indices, np.zeros(0)

    # Retrieval list of classes for this sample.

    retrieved_classes = np.argsort(scores)[::-1]

    # class_rankings[top_scoring_class_index] == 0 etc.

    class_rankings = np.zeros(num_classes, dtype=np.int)

    class_rankings[retrieved_classes] = range(num_classes)

    # Which of these is a true label?

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

    retrieved_class_true[class_rankings[pos_class_indices]] = True

    # Num hits for every truncated retrieval list.

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    # Precision of retrieval list truncated at each hit, in order of pos_labels.

    precision_at_hits = (

        retrieved_cumulative_hits[class_rankings[pos_class_indices]] /

        (1 + class_rankings[pos_class_indices].astype(np.float)))

    return pos_class_indices, precision_at_hits



# All-in-one calculation of per-class lwlrap.





def calculate_per_class_lwlrap(truth, scores):

    """Calculate label-weighted label-ranking average precision.



    Arguments:

      truth: np.array of (num_samples, num_classes) giving boolean ground-truth

        of presence of that class in that sample.

      scores: np.array of (num_samples, num_classes) giving the classifier-under-

        test's real-valued score for each class for each sample.



    Returns:

      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each

        class.

      weight_per_class: np.array of (num_classes,) giving the prior of each

        class within the truth labels.  Then the overall unbalanced lwlrap is

        simply np.sum(per_class_lwlrap * weight_per_class)

    """

    assert truth.shape == scores.shape

    num_samples, num_classes = scores.shape

    # Space to store a distinct precision value for each class on each sample.

    # Only the classes that are true for each sample will be filled in.

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):

        pos_class_indices, precision_at_hits = (

            _one_sample_positive_class_precisions(scores[sample_num, :],

                                                  truth[sample_num, :]))

        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

            precision_at_hits)

    labels_per_class = np.sum(truth > 0, axis=0)

    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    # Form average of each column, i.e. all the precisions assigned to labels in

    # a particular class.

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /

                        np.maximum(1, labels_per_class))

    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

    #                = np.sum(per_class_lwlrap * weight_per_class)

    return per_class_lwlrap, weight_per_class



# Calculate the overall lwlrap using sklearn.metrics function.





def calculate_overall_lwlrap_sklearn(truth, scores):

    """Calculate the overall lwlrap using sklearn.metrics.lrap."""

    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.

    sample_weight = np.sum(truth > 0, axis=1)

    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(

        truth[nonzero_weight_sample_indices, :] > 0,

        scores[nonzero_weight_sample_indices, :],

        sample_weight=sample_weight[nonzero_weight_sample_indices])

    return overall_lwlrap





# Accumulator object version.



class lwlrap_accumulator(object):

    """Accumulate batches of test samples into per-class and overall lwlrap."""



    def __init__(self):

        self.num_classes = 0

        self.total_num_samples = 0



    def accumulate_samples(self, batch_truth, batch_scores):

        """Cumulate a new batch of samples into the metric.



        Args:

          truth: np.array of (num_samples, num_classes) giving boolean

            ground-truth of presence of that class in that sample for this batch.

          scores: np.array of (num_samples, num_classes) giving the

            classifier-under-test's real-valued score for each class for each

            sample.

        """

        assert batch_scores.shape == batch_truth.shape

        num_samples, num_classes = batch_truth.shape

        if not self.num_classes:

            self.num_classes = num_classes

            self._per_class_cumulative_precision = np.zeros(self.num_classes)

            self._per_class_cumulative_count = np.zeros(self.num_classes,

                                                        dtype=np.int)

        assert num_classes == self.num_classes

        for truth, scores in zip(batch_truth, batch_scores):

            pos_class_indices, precision_at_hits = (

                _one_sample_positive_class_precisions(scores, truth))

            self._per_class_cumulative_precision[pos_class_indices] += (

                precision_at_hits)

            self._per_class_cumulative_count[pos_class_indices] += 1

        self.total_num_samples += num_samples



    def per_class_lwlrap(self):

        """Return a vector of the per-class lwlraps for the accumulated samples."""

        return (self._per_class_cumulative_precision /

                np.maximum(1, self._per_class_cumulative_count))



    def per_class_weight(self):

        """Return a normalized weight vector for the contributions of each class."""

        return (self._per_class_cumulative_count /

                float(np.sum(self._per_class_cumulative_count)))



    def overall_lwlrap(self):

        """Return the scalar overall lwlrap for cumulated samples."""

        return np.sum(self.per_class_lwlrap() * self.per_class_weight())
# import tensorflow.keras as keras

# from tensorflow.keras import layers

# from tensorflow.keras.callbacks import *

# from tensorflow.keras.layers import *

# from tensorflow.keras.optimizers import *

# from tensorflow.keras.losses import *

# from tensorflow.keras.models import Model



from keras import layers

from keras.callbacks import *

from keras.layers import *

from keras.optimizers import *

from keras.losses import *

from keras.models import Model





# Based on https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py

def block1(x, filters, kernel_size=3, stride=1,

           conv_shortcut=True, name=None):

    """A residual block.

    # Arguments

        x: input tensor.

        filters: integer, filters of the bottleneck layer.

        kernel_size: default 3, kernel size of the bottleneck layer.

        stride: default 1, stride of the first layer.

        conv_shortcut: default True, use convolution shortcut if True,

            otherwise identity shortcut.

        name: string, block label.

    # Returns

        Output tensor for the residual block.

    """

    bn_axis = 3



    if conv_shortcut is True:

        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,

                                 name=name + '_0_conv')(x)

        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                             name=name + '_0_bn')(shortcut)

    else:

        shortcut = x



    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                  name=name + '_1_bn')(x)

    x = layers.PReLU(name=name + '_1_prelu')(x)



    x = layers.Conv2D(filters, kernel_size, padding='SAME',

                      name=name + '_2_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                  name=name + '_2_bn')(x)

    x = layers.PReLU(name=name + '_2_prelu')(x)



    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                  name=name + '_3_bn')(x)



    x = layers.Add(name=name + '_add')([shortcut, x])

    x = layers.PReLU(name=name + '_out')(x)

    

    return x





def stack1(x, filters, blocks, stride1=2, name=None):

    """A set of stacked residual blocks.

    # Arguments

        x: input tensor.

        filters: integer, filters of the bottleneck layer in a block.

        blocks: integer, blocks in the stacked blocks.

        stride1: default 2, stride of the first layer in the first block.

        name: string, stack label.

    # Returns

        Output tensor for the stacked blocks.

    """

    x = block1(x, filters, stride=stride1, name=name + '_block1')

    for i in range(2, blocks + 1):

        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))

    return x





def ResNetlike(input_shape, num_classes):

    

    use_bias = False

    num_blocks = 1

    strides = (2, 4)

    

    input_layer = layers.Input(input_shape)

    # x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input_layer)

    x = layers.Conv2D(64, (3, 7), strides=2, use_bias=use_bias, name='conv1_conv')(input_layer)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                      name='conv1_bn')(x)

    x = layers.PReLU(name='conv1_prelu')(x)

    

    x = stack1(x, 64, num_blocks, stride1=1, name='conv2')

    x = stack1(x, 64, num_blocks, stride1=strides, name='conv3')

    x = stack1(x, 64, num_blocks, stride1=strides, name='conv4')

    

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    x = layers.Dense(num_classes, activation='softmax', name='probs')(x)

    

    model = Model(input_layer, x, name='resnetlike')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder





# Inspired by https://www.kaggle.com/yekenot/pooled-gru-fasttext

class LwlRapEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()

        self.interval = interval

        self.X_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = calculate_overall_lwlrap_sklearn(self.y_val, y_pred)

            print("\n LWLRAP - epoch: {} - score: {:.4f} \n".format(epoch +1, score))

        return





# Based on https://www.kaggle.com/voglinio/keras-2d-model-5-fold-log-specgram-curated-only

def create_unique_labels(all_labels):

    label_dict = {}

    all_labels_set = []

    first_labels_set = []

    for labs in all_labels:

        lab = labs.split(',')

        for l in lab:

            if l in label_dict:

                label_dict[l] = label_dict[l]  + 1

            else:

                label_dict[l]= 0



        all_labels_set.append(set(lab))

        first_labels_set.append(lab[0])

    classes = list(label_dict.keys())

    

    return label_dict, classes, all_labels_set, first_labels_set





label_dict, classes, all_labels_set, first_labels_set = create_unique_labels(train_df.labels)

binarize = MultiLabelBinarizer(classes=classes)

y_cat = binarize.fit_transform(all_labels_set)
RUN_NAME = 'ResNetLike_mel'

BATCH_SIZE = 32

NUM_EPOCHS = 35

NFOLDS = 5

SEED = 1337

LOAD_MODEL = False

TRAIN_MODEL = True

DEBUG = False

SAVE_SUBMISSION = True





# y_cat for categorical_crossentropy

# y_train.values for binary_crossentropy

labels_set = y_cat   

kfold = KFold(n_splits=NFOLDS, random_state=SEED)

num_classes = 80

bn_axis = 3



oof_train = np.zeros((X_train.shape[0], num_classes))

oof_test = np.zeros((X_test.shape[0], num_classes, NFOLDS))

valid_scores = []





# KFold training

for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(X_train)):

    X_tr, y_tr = X_train[train_idx], labels_set[train_idx]

    X_val, y_val = X_train[valid_idx], labels_set[valid_idx]

    print(X_tr.shape, y_tr.shape)

    print(X_val.shape, y_val.shape)

    

    checkpoint_name = '{}_{}.h5'.format(RUN_NAME, fold_idx)

    lwl_callback = LwlRapEvaluation(validation_data=(X_val, y_val))

    

    callbacks_list = [

        ModelCheckpoint(checkpoint_name, save_best_only=True, save_weights_only=True),

        ReduceLROnPlateau(patience=5, factor=0.2),

        EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True),

        lwl_callback,

    ]

    

    model = ResNetlike(X_train.shape[1:], y_train.shape[-1])

    # model = BaselineModel(X_train.shape[1:])

    # model.summary()

    if LOAD_MODEL:

        print('loading: {}'.format(checkpoint_name))

        model.load_weights(checkpoint_name)

    if TRAIN_MODEL:

        print('training model...')

        model.fit(

            X_tr, y_tr, 

            validation_data=(X_val, y_val),

            batch_size=BATCH_SIZE,

            epochs=NUM_EPOCHS,

            verbose=1,

            callbacks=callbacks_list)

    print('loading best weights from current fold')

    model.load_weights(checkpoint_name)

    

    val_pred = model.predict(X_val, batch_size=BATCH_SIZE)

    oof_train[valid_idx, :] = val_pred

    oof_test[:, :, fold_idx] = model.predict(X_test, batch_size=BATCH_SIZE)

    

    val_lwlrap = calculate_overall_lwlrap_sklearn(y_val, val_pred)

    valid_scores.append(val_lwlrap)

    print("lwlrap fold: {:.4f}".format(val_lwlrap))

    # break

    

    

oof_lwl = calculate_overall_lwlrap_sklearn(labels_set, oof_train)

print('OOF LWLRAP: {:.4f}'.format(oof_lwl))
oof_test_mean = oof_test.mean(axis=-1)

print(oof_test_mean.shape)



sort_idx = np.argsort(classes).astype(int)

oof_test_mean_sorted = oof_test_mean[:, sort_idx]

sample_submission.iloc[:, 1:] =  oof_test_mean_sorted

if SAVE_SUBMISSION:

    # sample_submission.to_csv('{}_lwl_{:.4f}.csv'.format(RUN_NAME, oof_lwl, index=False))

    sample_submission.to_csv('submission.csv', index=False)

    print(sample_submission.shape)

    

sample_submission.head()