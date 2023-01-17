# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%pylab inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import style; style.use('ggplot')

# Read the basic libraries (similar start as in Kaggle kernels)

import time # for timing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random # Random generators

import numpy as np

import pandas as pd # Pandas dataframe

import matplotlib.pyplot as plt

import re # Text cleaning

import nltk # Text processing

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from bs4 import BeautifulSoup # Text cleaning

import tensorflow as tf # Tensorflow

from tensorflow.keras import preprocessing # Text preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer # Text preprocessing

from tensorflow.keras.preprocessing.sequence import pad_sequences # Text preprocessing

from tensorflow.keras.models import Sequential # modeling neural networks

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, SpatialDropout1D, LSTM

from tensorflow.keras.initializers import Constant

from tensorflow.keras import optimizers, metrics # Neural Network

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

random.seed(10) # Set seed for the random generators

print(f"Tensorflow version: {tf.__version__}")
train = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv')

test = pd.read_csv('/kaggle/input/kuc-hackathon-winter-2018/drugsComTest_raw.csv')
train.head()
list(train)
train.values.shape[0], test.values.shape[0],

train.values.shape[0]/test.values.shape[0]
print("Train shape :" ,train.shape)

print("Test shape :",test.shape)
train.rating.hist(color = 'skyblue')

plt.title('Distribution of Ratings')

plt.xlabel('Rating')

plt.xticks([i for i in range(1,11)]);
from tensorflow.keras.preprocessing.text import Tokenizer

# For Train Data

samples = train['review']

tokenizer =Tokenizer(num_words = 5000)

tokenizer.fit_on_texts(samples)

# For Test Data

test_samples = test['review']

test_tokenizer =Tokenizer(num_words = 5000)

test_tokenizer.fit_on_texts(test_samples)



# Convert text to sequences for Train Data

sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))





# Convert text to sequences for Test Data

test_sequences = test_tokenizer.texts_to_sequences(test_samples)

test_word_index = test_tokenizer.word_index

print('Found %s unique tokens.' % len(test_word_index))
from tensorflow.keras.preprocessing.sequence import pad_sequences 



data = pad_sequences(sequences, maxlen=200)

data.shape



test_data = pad_sequences(test_sequences, maxlen=200)

test_data.shape
# Categorize labels for Train Data

labels = train ['rating'].values

labels = 1.0 * (labels >= 6 ) + 1.0*(labels >= 4)





# Categorize labels for Test Data

test_labels = test ['rating'].values

test_labels = 1.0 * (test_labels >= 6 ) + 1.0*(test_labels >= 4)





from tensorflow.keras.utils import to_categorical



# For train and validation 

labels = to_categorical(np.asarray(labels))#

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)



# For Test data

test_labels = to_categorical(np.asarray(test_labels))

print('Shape of data tensor:', test_data.shape)

print('Shape of label tensor:', test_labels.shape)
VALIDATION_SPLIT = 0.25

# split the data into a training set and a validation set

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]

y_val = labels[-nb_validation_samples:]
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding

from tensorflow.keras.models import Model

from keras import regularizers





embedding_layer = Embedding(5000,

                            100,

                            input_length=200,

                            trainable=True)





sequence_input = Input(shape=(200,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = GlobalMaxPooling1D()(x)

x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.05))(x)

preds = Dense(3, activation='softmax')(x)





model1 = Model(sequence_input, preds)

model1.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])





model1.summary()
history = model1.fit(x_train, y_train,

          batch_size=32,

          epochs=10,

          verbose=0,

          validation_data=(x_val, y_val))


acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

e = arange(len(acc)) + 1



plot(e, acc, label = 'train')

plot(e, val_acc, label = 'validation')

title('Training and validation accuracy')

xlabel('Epoch')

grid()

legend()



figure()



plot(e, loss, label = 'train')

plot(e, val_loss, label = 'validation')

title('Training and validation loss')

xlabel('Epoch')

grid()

legend()



show()
y_pred = argmax(model1.predict(x_val), axis = 1)

y_true = argmax(y_val, axis = 1)
cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
# Calculate the cohen's kappa, both with linear and quadratic weights

k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")
y_pred = argmax(model1.predict(test_data), axis = 1)

y_true = argmax(test_labels, axis = 1)
cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
# Calculate the cohen's kappa, both with linear and quadratic weights

k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")
model = Sequential()

model.add(Embedding(5000, 128, input_length=200))

model.add(SpatialDropout1D(0.1))

model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))

model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model.fit(x_train, y_train,

          batch_size=32,

          epochs=6,

          verbose=1,  

          validation_data=(x_val, y_val))
acc = history2.history['acc']

val_acc = history2.history['val_acc']

loss = history2.history['loss']

val_loss = history2.history['val_loss']

e = arange(len(acc)) + 1



plot(e, acc, label = 'train')

plot(e, val_acc, label = 'validation')

title('Training and validation accuracy')

xlabel('Epoch')

grid()

legend()



figure()



plot(e, loss, label = 'train')

plot(e, val_loss, label = 'validation')

title('Training and validation loss')

xlabel('Epoch')

grid()

legend()



show()
y_pred = argmax(model.predict(x_val), axis = 1)

y_true = argmax(y_val, axis = 1)
cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")
y_pred = argmax(model.predict(test_data), axis = 1)

y_true = argmax(test_labels, axis = 1)
cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
# Calculate the cohen's kappa, both with linear and quadratic weights

k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")