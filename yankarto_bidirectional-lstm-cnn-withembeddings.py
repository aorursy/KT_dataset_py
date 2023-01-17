import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Cleaning The texts using re library

import re



import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from collections import defaultdict

import string



from sklearn.feature_extraction.text import TfidfVectorizer





from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
from keras.preprocessing.text import Tokenizer

max_features = 50000



from keras.preprocessing.sequence import pad_sequences

maxlen = 100



import gensim

from gensim import corpora,models,similarities



import gc
train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

subm = pd.read_csv('../input/quora-insincere-questions-classification/sample_submission.csv')
from sklearn.model_selection import train_test_split



train, val_df = train_test_split(train, test_size=0.1, random_state=2018)
train1 = train["question_text"].fillna("_na_").values

val_X=val_df["question_text"].fillna("_na_").values

test1 = test["question_text"].fillna("_na_").values
embed_size = 300
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train1))



train1 = tokenizer.texts_to_sequences(train1)

test1 = tokenizer.texts_to_sequences(test1)

val_X=tokenizer.texts_to_sequences(val_X)





train1 = pad_sequences(train1, maxlen=maxlen)

test1 = pad_sequences(test1, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)



train_y = train['target'].values

val_y = val_df['target'].values
from zipfile import ZipFile

file_name='/kaggle/input/quora-insincere-questions-classification/embeddings.zip'

z=ZipFile(file_name)

print(z.namelist())
EMBEDDING_FILE=z.extract('paragram_300_sl999/paragram_300_sl999.txt')
import os

embedding_dict = {}

filename = os.path.join(os.getcwd()+'/paragram_300_sl999/paragram_300_sl999.txt')

with open(filename,encoding='utf8',errors='ignore') as f:

    for line in f:

        line = line.split()

        token = line[0]

        try:

            coefs = np.asarray(line[1:], dtype='float32',)

            embedding_dict[token] = coefs

        except:

            pass

print('The embedding dictionary has {} items'.format(len(embedding_dict)))

       

word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_1 = np.zeros(shape=[nb_words, embed_size])

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_dict.get(word)

    if embedding_vector is not None: embedding_matrix_1[i] = embedding_vector



del embedding_dict; gc.collect() 
S_DROPOUT = 0.4

DROPOUT = 0.1
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, concatenate

from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn import metrics

import gc
import tensorflow as tf

import random as rn



batch_size=512



np.random.seed(40)

rn.seed(40)

tf.random.set_seed(40)



from keras.layers import Input, Dense, Embedding,LSTM

from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout

from keras.models import Model

   

# def create_lstm():

input = Input(shape=(maxlen,))

   

# Embedding layer has fixed weights, so set 'trainable' to False

x = Embedding(nb_words, embed_size, weights=[embedding_matrix_1], trainable=False)(input)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

   

#     return model

model.fit(train1, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_val_lstm_y = model.predict([val_X], batch_size=1024, verbose=1)
pred_test_lstm_y = model.predict([test1], batch_size=1024, verbose=1)
pred_val_y = pred_val_lstm_y   # two random numbers

pred_test_y = pred_test_lstm_y



thresholds = []

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 1)

    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))

    thresholds.append([thresh, res])

    print("F1 score at threshold {0} is {1}".format(thresh, res))

    

thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]

print("Best threshold: ", best_thresh)
import time



from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D



filter_sizes = [1,2,3,5]

num_filters = 36



inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix_1])(inp)

x = SpatialDropout1D(S_DROPOUT)(x)

x = Reshape((maxlen, embed_size, 1))(x)



maxpool_pool = []

for i in range(len(filter_sizes)):

    conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),

                                 kernel_initializer='he_normal', activation='elu')(x)

    maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))



z = Concatenate(axis=1)(maxpool_pool)   

z = Flatten()(z)

z = Dropout(DROPOUT)(z)



outp = Dense(1, activation="sigmoid")(z)



model = Model(inputs=inp, outputs=outp)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
model.fit(train1, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_val_cnn_y = model.predict([val_X], batch_size=1024, verbose=1)
pred_test_cnn_y = model.predict([test1], batch_size=1024, verbose=1)
# del word_index, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
pred_val_y = 0.6 * pred_val_lstm_y + 0.4 * pred_val_cnn_y  # two random numbers

pred_test_y = 0.6 * pred_test_lstm_y + 0.4 * pred_test_cnn_y



thresholds = []

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))

    thresholds.append([thresh, res])

    print("F1 score at threshold {0} is {1}".format(thresh, res))

    

thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]

print("Best threshold: ", best_thresh)
pred_test_y = (pred_test_y > best_thresh).astype(int)

# out_df = pd.DataFrame({"qid":test1["qid"].values})

subm['prediction'] = pred_test_y

subm.to_csv("submission.csv", index=False)