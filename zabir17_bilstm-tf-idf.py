# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from sklearn.utils import shuffle

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

subm = pd.read_csv('../input/sample_submission.csv')
train.head()
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

num_classes = len(label_cols) 

y_train = train[label_cols].values

print(num_classes)

#train.head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(max_features=200,tokenizer=tokenize,sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', lowercase=False , ngram_range=(1, 2), stop_words='english')



X_train = tfidf.fit_transform(train[COMMENT]).toarray()

X_test = tfidf.transform(test[COMMENT])
#Deep learning implementation



from keras import utils

from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout, ActivityRegularization,Activation

from keras.models import Sequential

from keras.layers import BatchNormalization

from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras import optimizers

from collections import defaultdict

import gzip

import itertools
import os

from keras.layers import Input,Bidirectional,Activation,Conv1D,GRU

from keras.callbacks import Callback

from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten

from keras.preprocessing import text, sequence

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Model

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# Build the keras model



def build_model(X_train,num_classes):

    model = Sequential()

    model.add(Dense(64,input_shape=(X_train.shape[1],)))   

#    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],)))



#    model.add(SpatialDropout1D(0.3))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

#    model.add(Flatten())

#    model.add(Bidirectional(LSTM(64, return_sequences=True)))

#    model.add(Activation('relu'))

#    model.add(BatchNormalization())

    model.add(Dense(128))

    model.add(Dropout(0.3))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

#    model.add(GlobalMaxPooling1D())

    model.add(Dense(64))

    model.add(Dropout(0.3))

    model.add(Activation('relu'))

    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['acc'])

    return model
modelDL = build_model(X_train,num_classes)

modelDL.summary()
# From Deep Learning with Python book

def make_history_plot(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(acc) + 1)



    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b-', color='green',label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b-', color='green',label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show()
# Train the model, record history

#print(y_train)

batch_size=32

historyDL = modelDL.fit(X_train, y_train,

                    epochs=10,

                    batch_size=batch_size,

                    verbose=1,

                    validation_split=0.1)

import matplotlib.pyplot as plt

make_history_plot(historyDL)
y_test = modelDL.predict([X_test], batch_size=1024, verbose=1)

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission[label_cols] = y_test

sample_submission.to_csv('submission.csv', index=False)