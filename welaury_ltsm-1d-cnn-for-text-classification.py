# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !pip install -U tensorflow-gpu==1.15.2

# !pip install keras==2.3.0

# !pip install scikit-learn==0.22.1

!pip install pymystem3

# !pip install numpy==1.18.2

# !pip install pandas==0.25.3

# !pip install seaborn==0.10.0

# !pip install matplotlib==3.2.1

!pip install missingno==0.4.2
# Load libraries

import tensorflow

print(tensorflow.__version__) # make sure the version of tensorflow

import numpy as np # for scientific computing

import pandas as pd # for data analysis

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns # for data visualization

import missingno as msno # for missing data visualization

import collections

import nltk

import codecs

import string

import re

from tqdm import tqdm

from collections import defaultdict

from collections import Counter 

from keras.initializers import Constant

from keras.callbacks import ModelCheckpoint

from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from keras.optimizers import Adam

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, SpatialDropout1D, Embedding, LSTM, Conv1D, GlobalMaxPooling1D

from tensorflow.keras.layers import Dropout

plt.style.use('ggplot')

np.random.seed(42) # set the random seeds
train = pd.read_csv('/kaggle/input/kinopoisk-remove-stopwords/train_proc.csv', low_memory = False)

test = pd.read_csv('/kaggle/input/kinopoisk-remove-stopwords/test_proc.csv', low_memory = False)

train['target'] = train[['positive', 'negative', 'neutral']].idxmax(axis=1)



train = train[['preprocess_text', 'target']]
train.head()
test.head()
sns.countplot(x='target', data=train)
train = train[train['preprocess_text'] != '']
max_len = 881

max_fatures = 500000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(train['preprocess_text'].values)



train_converted = tokenizer.texts_to_sequences(train['preprocess_text'].values)

train_converted = pad_sequences(train_converted, maxlen=max_len)



test = tokenizer.texts_to_sequences(test['preprocess_text'].values)

test = pad_sequences(test, maxlen=max_len)
target_converted = pd.get_dummies(train['target']).values
pd.get_dummies(train['target'])
print('The shape of train data :', train_converted.shape)

print('The shape of test data :', test.shape)

print('The shape of target of the training :', target_converted.shape)
X_train, X_test, Y_train, Y_test = train_test_split(train_converted, target_converted, test_size = 0.1, random_state = 42)



train_converted_shape = train_converted.shape[1]

del(train_converted)

del(target_converted)



validation_size = 5000

X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]



print('The shape of train data :', X_train.shape)

print('The shape of labels of train data :', Y_train.shape)

print('The shape of test data :', X_test.shape)

print('The shape of test label data :', Y_test.shape)
# embed_dim = 256

# lstm_out = 196

# batch_size = 4

# EPOCHS = 10



# model = Sequential()

# model.add(Embedding(max_fatures, embed_dim,input_length = train_converted_shape))

# model.add(SpatialDropout1D(0.3))

# model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3))

# model.add(Dense(3,activation='softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = [tensorflow.keras.metrics.AUC()])

# print(model.summary())
# history = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size=batch_size, 

#                     validation_data=(X_validate, Y_validate), verbose = 2)



# train_acc = history.history['auc']

# test_acc = history.history['val_auc']

# x = np.arange(len(train_acc))

# plt.plot(x, train_acc, label = 'train auc')

# plt.plot(x, test_acc, label = 'test auc')

# plt.title('Train and validation auc')

# plt.xlabel('Number of epochs')

# plt.ylabel('auc')

# plt.legend()
NUM_FILTERS = 1024

NUM_WORDS = 7

embed_dim = 256

batch_size = 32

EPOCHS = 1



model = Sequential()

model.add(Embedding(max_fatures, embed_dim, input_length = train_converted_shape))

model.add(SpatialDropout1D(0.2))

model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))

model.add(GlobalMaxPooling1D())

#model.add(Dense(256, activation="linear"))

model.add(Dense(3, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=[tensorflow.keras.metrics.AUC()])

print(model.summary())
history = model.fit(X_train, Y_train, batch_size=batch_size,

                    epochs=EPOCHS, validation_data=(X_validate, Y_validate),

                    callbacks=[ModelCheckpoint('bm.h5', monitor='val_loss', verbose=1, save_best_only=True)])



# train_acc = history.history['auc']

# test_acc = history.history['val_auc']

# x = np.arange(len(train_acc))

# plt.plot(x, train_acc, label = 'train auc')

# plt.plot(x, test_acc, label = 'test auc')

# plt.title('Train and validation auc')

# plt.xlabel('Number of epochs')

# plt.ylabel('auc')

# plt.legend()
submit = pd.read_csv('/kaggle/input/ml-guild-classification-task/sample_submission.csv')

submit.head()
pred = model.predict(test, batch_size=8)

pred[:10]
submit[['negative', 'neutral', 'positive']] = pred
submit[['positive', 'negative', 'neutral']] = submit[['positive', 'negative', 'neutral']].apply(round)
submit[(submit.negative == 0) & (submit.neutral == 0) & (submit.positive == 0)]
submit.to_csv('submission.csv', index=False)