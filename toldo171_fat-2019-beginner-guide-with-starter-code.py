import time

start_time = time.time()



# Data analysis and wrangling

import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import IPython

import IPython.display

import librosa

import librosa.display

import random

from tqdm import tqdm_notebook

from fastai import *

from fastai.vision import *

from fastai.vision.data import *

from fastai.imports import *

from fastai.callback import *

from fastai.callbacks import *



# Machine learning

from sklearn import preprocessing

import sklearn.metrics

from sklearn.metrics import label_ranking_average_precision_score



# File handling

from pathlib import Path

import gc

import os

print(os.listdir("../input"))
# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU

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





# Wrapper for fast.ai library

def lwlrap(scores, truth, **kwargs):

    score, weight = calculate_per_class_lwlrap(to_np(truth), to_np(scores))

    return torch.Tensor([(score * weight).sum()])
training_curated_df = pd.read_csv("../input/train_curated.csv")

training_noisy_df = pd.read_csv("../input/train_noisy.csv")

training_df = [training_curated_df, training_noisy_df]

testing_df = pd.read_csv('../input/sample_submission.csv')
Path('trn_curated').mkdir(exist_ok=True, parents=True)

Path('trn_noisy').mkdir(exist_ok=True, parents=True)

Path('test').mkdir(exist_ok=True, parents=True)
# preview the data

training_curated_df.head()
# preview the data

training_noisy_df.head()
training_curated_df.info()

print('_'*40)

training_noisy_df.info()
labels_curated = training_curated_df['labels'].unique()

print(labels_curated.shape)

print('_'*40)

print(labels_curated)
labels_noisy = training_noisy_df['labels'].unique()

print(labels_noisy.shape)

print('_'*40)

print(labels_noisy)
training_curated_df.describe()
#EasyDict allows to access dict values as attributes (works recursively). A Javascript-like properties dot notation for python dicts.

#It is mandatory in order to use the library below

# Special thanks to https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py

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
#Thanks to https://github.com/daisukelab/ml-sound-classifier

def read_audio(conf, pathname, trim_long_data):

    y, sr = librosa.load(pathname, sr=conf.sampling_rate) #Loads an audio file as a floating point time series. This functions samples the sound

    # trim silence

    if 0 < len(y): # workaround: 0 length causes error

        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # make it unified length to conf.samples

    if len(y) > conf.samples: # long enough

        if trim_long_data:

            y = y[0:0+conf.samples]

    else: # pad blank

        padding = conf.samples - len(y)    # add padding at both ends

        offset = padding // 2

        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    return y



def audio_to_melspectrogram(conf, audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=conf.sampling_rate,

                                                 n_mels=conf.n_mels,

                                                 hop_length=conf.hop_length,

                                                 n_fft=conf.n_fft,

                                                 fmin=conf.fmin,

                                                 fmax=conf.fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32) #Returns an 128 x L array corresponding to the spectrogram of the sound (L = 128*n° of s)

    return spectrogram



def melspectrogram_to_delta(mels):

    return librosa.feature.delta(mels)



def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 

                             sr=conf.sampling_rate, hop_length=conf.hop_length,

                            fmin=conf.fmin, fmax=conf.fmax)

    plt.colorbar(format='%+2.0f dB')

    plt.title(title)

    plt.show()



def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):

    x = read_audio(conf, pathname, trim_long_data)

    mels = audio_to_melspectrogram(conf, x)

    if debug_display:

        delta = melspectrogram_to_delta(mels)

        delta_squared = melspectrogram_to_delta(delta)

        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))

        show_melspectrogram(conf, mels)

        show_melspectrogram(conf, delta)

        show_melspectrogram(conf, delta_squared)

    return mels



conf = EasyDict()

conf.sampling_rate = 44100

conf.duration = 2

conf.hop_length = 347 * conf.duration # to make time steps 128

conf.fmin = 20

conf.fmax = conf.sampling_rate // 2

conf.n_mels = 128

conf.n_fft = conf.n_mels * 20

conf.samples = conf.sampling_rate * conf.duration
# example

path = '../input/train_curated/0006ae4e.wav'

x = read_audio(conf, path, trim_long_data=False)

print(x)

print('_'*40)

print(audio_to_melspectrogram(conf, x))

print(audio_to_melspectrogram(conf, x).shape)

x1 = read_as_melspectrogram(conf, path, trim_long_data=False, debug_display=True)
"""

The mono_to_color function takes as an input the spectrogram of our sound (list of array, see above). 

It stacks it three times, so that it has the same shape as a classic RGB image.

Then it standardize the array (take a matrix and change it so that its mean is equal to 0 and variance is 1). This improves performance.

Then it normalizes each value between 0 and 255 (gray scale). 

"""



def mels_preprocessing(X1, X2, X3, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    # Stack X as [X,X,X]

    X = np.stack([X1, X2, X3], axis=-1)



    # Standardize

    mean = mean or X.mean()

    std = std or X.std()

    #Standardization. Xstd has 0 mean and 1 variance

    Xstd = (X - mean) / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Scale to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V



def convert_wav_to_image(df, source, img_dest):

    X = []

    for i, row in tqdm_notebook(df.iterrows()):

        x1 = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)

        x2 = melspectrogram_to_delta(x1)

        x3 = melspectrogram_to_delta(x2)

        x_preprocessed = mels_preprocessing(x1, x2, x3)

        X.append(x_preprocessed)

    return df, X
training_curated_df, X_train_curated = convert_wav_to_image(training_curated_df, source=Path('../input/train_curated'), img_dest=Path('trn_curated'))

testing_df, X_test = convert_wav_to_image(testing_df, source=Path('../input/test'), img_dest=Path('test'))



print(f"Finished data conversion at {(time.time()-start_time)/3600} hours")
for i in range(0,6):

    a = np.asarray(X_train_curated[i:i+1])

    a = np.squeeze(a)

    print(a.shape)

    plt.imshow(a)

    plt.show()
CUR_X_FILES, CUR_X = list(training_curated_df.fname.values), X_train_curated



def open_fat2019_image(fn, convert_mode, after_open)->Image:

    # open

    idx = CUR_X_FILES.index(fn.split('/')[-1])

    x = PIL.Image.fromarray(CUR_X[idx])

    # crop

    time_dim, base_dim = x.size

    crop_x = 0

    #crop_x = random.randint(0, time_dim - base_dim)

    x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])    

    # standardize

    return Image(pil2tensor(x, np.float32).div_(255))



vision.data.open_image = open_fat2019_image
#Batch size --> How many images are trained at one time. Lower it if you run out of memory

bs = 64

#Image size. Square images makes the learning process faster. We can increase the size of the images once our model is stable, in order to improve accuracy.

size = 128



#Performing data augmentation

tfms = get_transforms(do_flip=False, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)



#We put the transformed image data into /kaggle/working because ../input is a read-only directory.

src = (ImageList.from_csv('/kaggle/working', '../input/train_curated.csv', folder='../input/train_curated')

       .split_by_rand_pct(0.2).label_from_df(label_delim=','))



#Creates a databunch, because our cnn_learner below needs a databunch.

data = src.transform(tfms, size=size).databunch(bs=bs).normalize()
data.show_batch(4)
arch = models.resnet18



learn = cnn_learner(data, arch, pretrained=False, metrics=[lwlrap], wd = 0.1, ps = 0.5)



learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-2)

learn.save('first-attempt-128')
learn.unfreeze()

learn.fit_one_cycle(1)
learn.lr_find()

learn.recorder.plot()
learn.fit(20, slice(2e-3, 2e-4))

learn.save('second-attempt-128')
learn.lr_find()

learn.recorder.plot()
learn.fit(20, slice(1e-3, 1e-4))

learn.save('third-attempt-128')
size = 256



#Creates a databunch, because our cnn_learner below needs a databunch.

data = src.transform(tfms, size=size).databunch(bs=bs).normalize(imagenet_stats)
#Replace with 256x256 databunch

learn.data = data

#Freeze the model

learn.freeze()

#Plot lr_find()

learn.lr_find()

learn.recorder.plot()
learn.fit(5, 3e-3)

learn.save('first-attempt-256')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit(7, slice(1e-3, 1e-4))

learn.save('second-attempt-256')
learn.recorder.plot_losses()
learn.fit_one_cycle(20, slice(5e-4, 5e-5), callbacks=[SaveModelCallback(learn, monitor='lwlrap', mode='max')])
learn.export()
CUR_X_FILES, CUR_X = list(testing_df.fname.values), X_test



test = ImageList.from_csv(Path('/kaggle/working'), Path('../input/sample_submission.csv'), folder=Path('../input/test'))

learn = load_learner(Path('/kaggle/working'), test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)



testing_df[learn.data.classes] = preds

testing_df.to_csv('submission.csv', index=False)

testing_df.head()