# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# TF Imports

import tensorflow as tf
# TF Imports

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, CuDNNLSTM, CuDNNGRU, Dropout, Bidirectional, Conv1D, Input

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate

from tensorflow.python.keras.layers.embeddings import Embedding

from tensorflow.python.keras.preprocessing import sequence

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler

# Numpy

import numpy

numpy.random.seed(1331)

# Pandas

import pandas as pd

# Sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score

# Visualizations

import matplotlib.pyplot as plt

%matplotlib inline

# Garbage Collector

import gc

import sys

# Hyperopt

from hyperopt import fmin, tpe, hp, anneal, Trials, space_eval

# Random

import random

# codecs + collections + csv

import codecs

import collections

import csv
tf.__version__
# load training data

TRAIN_DATA_FILE = "../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv"

j_df = pd.read_csv(TRAIN_DATA_FILE)



# Parameter settings

maxlen = 220

max_features = 200000



# create a tokenizer

token = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)

# fit tokenizer on data

token.fit_on_texts(j_df['comment_text'])

# get word index from tokenizer

word_index = token.word_index



# Memory Clean-up

del j_df

gc.collect()
# load validation data

VAL_DATA_FILE = "../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv"

j_df = pd.read_csv(VAL_DATA_FILE)

j_df.head(5)


X_val = sequence.pad_sequences(token.texts_to_sequences(j_df['translated']), maxlen=maxlen)

y_val = j_df['toxic']

print(X_val.shape)



# Memory Clean-up

del j_df

gc.collect()
embed_size = 100



EMBEDDING_FILES = [

    '../input/jigsaw-custom-word2vec-100d-5iter/custom_word2vec_100d_5iter.txt'

]



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 100))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix



embedding_matrix = np.concatenate([build_matrix(token.word_index, f) for f in EMBEDDING_FILES], axis=-1)

embedding_matrix = embedding_matrix[0:max_features,:]

embedding_matrix.shape
def sent_generator(TRAIN_DATA_FILE, chunksize, threshold, maxlen):

    reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)

    for df in reader:

        texts = df.iloc[:,1].astype(str)

        target = np.where(df.iloc[:,2]>threshold,1,0)

        sequences = token.texts_to_sequences(texts)

        data_train = sequence.pad_sequences(sequences, maxlen=maxlen)

        yield data_train, target
def test_data_prep():

    # load test data

    j_df = pd.read_csv("../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")

    X_test = j_df['translated'].astype(str)

    del j_df

    gc.collect()

    X_test = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=maxlen)

    return X_test



X_test = test_data_prep()
# Results from Hyperopt

drpt_amt = 0.30

lstm2_nrns = 23

epochs = 1

batches = 641
def build_model():

    # create model

    model = Sequential()

    model.add(Embedding(max_features, embed_size, input_length=maxlen, weights=[embedding_matrix], trainable=False))

    model.add(Dropout(drpt_amt))

    model.add(Bidirectional(CuDNNLSTM(lstm2_nrns)))

    model.add(Dropout(drpt_amt))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
checkpoint_predictions = []

weights = []

nb_epoch = epochs

batch_size = batches

threshold = 0.48

for model_idx in range(2):

    model = build_model()

    n_steps = (1209267) // batch_size

    threshold += 0.02

    for counter in range(nb_epoch):

        print('-------epoch: ',counter,'--------')

        scheduler = lambda _: 1e-3 * (0.55 ** counter)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model.fit_generator(sent_generator(TRAIN_DATA_FILE, batch_size, threshold, 220),

                            steps_per_epoch=n_steps, 

                            epochs=3, 

                            validation_data=(X_val, y_val),

                            callbacks=[callback])

        prediction = model.predict_proba(X_test).flatten()

        checkpoint_predictions.append(prediction)

        weights.append(2 ** counter)
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



test_df = pd.read_csv("../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")

submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'toxic': predictions

})

submission.to_csv('submission.csv', index=False)

submission.head()