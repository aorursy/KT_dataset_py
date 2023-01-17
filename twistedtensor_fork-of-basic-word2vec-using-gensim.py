!ls ../input/
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import gensim
import multiprocessing
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
# Remove all non-letter characters and make everything lowercase.
train['comment_text'] = train['comment_text'].str.replace('[^a-zA-Z]',' ').str.lower()
test['comment_text'] = test['comment_text'].str.replace('[^a-zA-Z]',' ').str.lower()
train['comment_text'].head(10)
train_split = train['comment_text'].str.split()
test_split = test['comment_text'].str.split()
train_lol = train_split.values.tolist()
test_lol = test_split.values.tolist()
# Gensim has support for multi-core systems
multiprocessing.cpu_count()
# I have no reason in mind to change the default word2vec parameters, so I will use the defaults
w2v = gensim.models.word2vec.Word2Vec(sentences=train_lol,workers=32)
from collections import defaultdict
vocab = defaultdict(int)
for k, v in w2v.wv.vocab.items():
    vocab[k]=v.index
max([v for k,v in vocab.items()])
train_ind = [[vocab[w] for w in train_split[i]] for i in range(len(train_split))]
test_ind = [[vocab[w] for w in test_split[i]] for i in range(len(test_split))]
from keras.preprocessing.sequence import pad_sequences
train_padded = pad_sequences(train_ind,maxlen=100,truncating='pre')
test_padded = pad_sequences(test_ind,maxlen=100,truncating='pre')
train_padded
train_padded.shape
emb_layer = w2v.wv.get_keras_embedding()
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GlobalMaxPool1D, BatchNormalization,Embedding
emb_layer
import logging
from keras.callbacks import Callback
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

inp = Input(shape=(maxlen,))
x = emb_layer(inp)
x = Bidirectional(LSTM(50, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = BatchNormalization()(x)
x = Dense(50, activation="relu")(x)
#x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

import keras.backend as K
def loss(y_true, y_pred):
     return K.binary_crossentropy(y_true, y_pred)
    
model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
from keras import callbacks
def schedule(ind):
    a = [0.002,0.003, 0.000]
    return a[ind]
lr = callbacks.LearningRateScheduler(schedule)

model.fit(x=train_padded,y=y,validation_split=.1,epochs=3,batch_size=64)

#model.fit(x=train_padded,y=y,validation_split=.1,epochs=3,batch_size=64)
test_pred = model.predict(x=test_padded)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission[list_classes] = test_pred
sample_submission.to_csv('LSTM-submission2.csv', index=False)

