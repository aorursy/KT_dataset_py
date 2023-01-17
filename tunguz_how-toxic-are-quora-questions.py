# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline  

import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv').fillna(' ')
test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv').fillna(' ')
test_qid = test['qid']
train_qid = train['qid']
train_target = train['target'].values

train_text = train['question_text']
test_text = test['question_text']

all_text = pd.concat([train_text, test_text])


embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embed_size = 300
max_features = 130000
max_len = 220

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_text = train_text.str.lower()
test_text = test_text.str.lower()
all_text = all_text.str.lower()
tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(all_text)
all_text = tk.texts_to_sequences(all_text)
train_text = tk.texts_to_sequences(train_text)
test_text = tk.texts_to_sequences(test_text)
train_pad_sequences = pad_sequences(train_text, maxlen = max_len)
test_pad_sequences = pad_sequences(test_text, maxlen = max_len)
all_pad_sequences = pad_sequences(all_text, maxlen = max_len)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
model = load_model("../input/bi-gru-lstm-cnn-poolings-fasttext/best_model.hdf5")

train_pred = model.predict(train_pad_sequences, batch_size = 1024, verbose = 1)
test_pred = model.predict(test_pad_sequences, batch_size = 1024, verbose = 1)
#all_pred = model.predict(all_pad_sequences, batch_size = 1024, verbose = 1)
train_pred.max()
test_pred.max()
toxic_predictions_train = pd.DataFrame(columns=list_classes, data=train_pred)
toxic_predictions_test = pd.DataFrame(columns=list_classes, data=test_pred)
toxic_predictions_train['question_text'] = train['question_text'].values
toxic_predictions_test['question_text'] = test['question_text'].values
toxic_predictions_train['qid'] = train_qid
toxic_predictions_test['qid'] = test_qid
toxic_predictions_train.head()
toxic_predictions_test.head()
toxic_predictions_train[list_classes].describe()
toxic_predictions_test[list_classes].describe()
print(toxic_predictions_train.sort_values(by=['toxic'], ascending=False)['question_text'].head(10).values)
print(toxic_predictions_train.sort_values(by=['severe_toxic'], ascending=False)['question_text'].head(10).values)
print(toxic_predictions_train.sort_values(by=['obscene'], ascending=False)['question_text'].head(10).values)
print(toxic_predictions_train.sort_values(by=['threat'], ascending=False)['question_text'].head(10).values)
print(toxic_predictions_train.sort_values(by=['insult'], ascending=False)['question_text'].head(10).values)
print(toxic_predictions_train.sort_values(by=['identity_hate'], ascending=False)['question_text'].head(10).values)