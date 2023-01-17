import copy

import gc

import glob

import os

import time



import cv2

import IPython

import IPython.display

import librosa

import librosa.display

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow.keras as keras

from joblib import Parallel, delayed

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

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

        

    

processor = DataProcessor()
FILENAME = train_root + train_df.fname[5]





sample_mel = processor.createMel(FILENAME, CONFIG)

sample_mfcc = processor.createMfcc(FILENAME, CONFIG)

print(sample_mel.shape)

print(sample_mfcc.shape)



idx_cut = 400

plt.imshow(sample_mel[:, :idx_cut], cmap='Spectral')

plt.title('Mel Spectrogram:')

plt.show()

plt.imshow(sample_mfcc[:, :idx_cut], cmap='Spectral')

plt.title('MFCC:')

plt.show()



# Mel sample range:

print(np.min(sample_mel), np.max(sample_mel))

# MFCC sample range:

print(np.min(sample_mfcc), np.max(sample_mfcc))
FILENAME  = train_root + train_df.fname[5]

NORMALIZE = True



sample_mel = processor.createMel(FILENAME, CONFIG, normalize=NORMALIZE)

sample_mfcc = processor.createMfcc(FILENAME, CONFIG, normalize=NORMALIZE)

print(sample_mel.shape)

print(sample_mfcc.shape)



idx_cut = 400

plt.imshow(sample_mel[:, :idx_cut], cmap='Spectral')

plt.title('Mel Spectrogram:')

plt.show()

plt.imshow(sample_mfcc[:, :idx_cut], cmap='Spectral')

plt.title('MFCC:')

plt.show()



# Mel sample range:

print(np.min(sample_mel), np.max(sample_mel))

# MFCC sample range:

print(np.min(sample_mfcc), np.max(sample_mfcc))
sample_idxs = np.random.choice(np.arange(0, len(train_df)), 5)





for i in sample_idxs:

    sample_prep, labels = processor.prepareSample(

        train_root,  # training set directory

        train_df.iloc[i, :],  # sample index from train_df 

        processor.createMel,  # function for data processing

        CONFIG, TRAINING_CONFIG,  # parameters for processing and training

        test_mode=False,

        proc_mode='split',  # indicate split into N sub-arrays

    )  # whether to labels are available

    print(sample_prep.max(), sample_prep.min())

    print(sample_prep.shape, labels.shape)

    NCOLS = 2

    NROWS = int(np.ceil(sample_prep.shape[0] / NCOLS))

    fig, ax = plt.subplots(NCOLS, NROWS, figsize=(20, 5))

    fig.suptitle('Sample: {}'.format(i))

    idx = 0

    for c in range(NCOLS):

        if NROWS > 1:

            for r in range(NROWS):

                if idx < sample_prep.shape[0]:

                    ax[c, r].imshow(sample_prep[idx], cmap='Spectral')

                    ax[c, r].set_title('class: {}'.format(labels[idx]))

                    idx += 1

        else:

            if idx < sample_prep.shape[0]:

                ax[c].imshow(sample_prep[idx], cmap='Spectral')

                ax[c].set_title('class: {}'.format(labels[idx]))

                idx += 1

    plt.show()
sample_idxs = np.random.choice(np.arange(0, len(train_df)), 5)





for i in sample_idxs:

    sample_prep, labels = processor.prepareSample(

        train_root,  # training set directory

        train_df.iloc[i, :],  # sample index from train_df 

        processor.createMel,  # function for data processing

        CONFIG, TRAINING_CONFIG,  # parameters for processing and training

        test_mode=False,

        proc_mode='resize'  # resize mode

    )  # whether to labels are available

    print(sample_prep.max(), sample_prep.min())

    print(sample_prep.shape, labels.shape)

    NCOLS = 1

    NROWS = 1

    fig, ax = plt.subplots(NCOLS, NROWS, figsize=(10, 3))

    fig.suptitle('Sample: {}, class: {}'.format(i, labels))

    ax.imshow(sample_prep[0], cmap='Spectral')

    plt.show()
sample_idxs = np.random.choice(np.arange(0, len(sample_submission)), 5)





for i in sample_idxs:

    sample_prep = processor.prepareSample(

        test_root,  # test set directory

        sample_submission.iloc[i, :],  # sample index from sample_submission 

        processor.createMel,  # function for data processing

        CONFIG, TRAINING_CONFIG,  # parameters for processing and training

        test_mode=True,  # indicate test mode

        proc_mode='resize'

    )  # whether to labels are available

    print(sample_prep.max(), sample_prep.min())

    print(sample_prep.shape, labels.shape)

    NCOLS = 1

    NROWS = int(np.ceil(sample_prep.shape[0] / NCOLS))

    fig, ax = plt.subplots(NCOLS, NROWS, figsize=(10, 3))

    fig.suptitle('Test Sample: {}'.format(i))

    ax.imshow(sample_prep[0], cmap='Spectral')

    plt.show()
output = Parallel(n_jobs=-3, verbose=1)(

    delayed(processor.prepareSample)(

        train_root, 

        train_df.iloc[f, :],

        processor.createMel,

        CONFIG,

        TRAINING_CONFIG,

        test_mode=False,

        proc_mode='resize',

    ) for f in range(100))  # change to number of samples in train data for full processing





X_train = np.array([x[0] for x in output])

y_train = np.array([x[1] for x in output])

y_train = pd.Series(y_train).str.get_dummies(sep=',')

print(X_train.shape, y_train.shape)
X_test = Parallel(n_jobs=-3, verbose=1)(

    delayed(processor.prepareSample)(

        test_root, 

        sample_submission.iloc[f, :],

        processor.createMel,

        CONFIG,

        TRAINING_CONFIG,

        test_mode=True,

        proc_mode='resize',

    ) for f in range(100))  # change to number of samples in test data for full processing





X_test = np.array(X_test)

print(X_test.shape)