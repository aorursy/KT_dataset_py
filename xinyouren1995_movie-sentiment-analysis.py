import pyprind

import pandas as pd

from string import punctuation

import re

import numpy as np

df = pd.read_csv('../input/IMDB Dataset.csv', encoding='utf-8')
from collections import Counter

counts = Counter()

pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrences')

for i,review in enumerate(df['review']):

    text=''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()

    df.loc[i,'review'] = text

    pbar.update()

    counts.update(text.split())



## Create a mapping

## Map each unique word to an integer

word_counts = sorted(counts, key=counts.get, reverse=True)

print(word_counts[:5])

word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}



mapped_reviews = []

pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to ints')

for review in df['review']:

    mapped_reviews.append([word_to_int[word] for word in review.split()])

    pbar.update()
## Define same-length sequences

## if sequence length < 200: left-pad with zeros

## if sequence length > 200: use the last 200 elements

sequence_length = 200 ## (Known as T in our RNN formulas)

sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for i, row in enumerate(mapped_reviews):

    review_arr = np.array(row)

    sequences[i, -len(row):] = review_arr[-sequence_length:]
X_train = sequences[:25000,:]

y_train = df.loc[:24999, 'sentiment'].values

y_train[y_train=='positive'] = 1

y_train[y_train=='negative'] = 0

X_test = sequences[25000:,:]

y_test = df.loc[25000:, 'sentiment'].values

y_test[y_test=='positive'] = 1

y_test[y_test=='negative'] = 0
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers

n_words = max(list(word_to_int.values())) + 1

mbed_size = 200

model = Sequential()

model.add(Embedding(n_words, mbed_size))

model.add(Bidirectional(LSTM(32, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model, show_shapes=True, show_layer_names=False, 

                 rankdir='LR').create(prog='dot', format='svg'))
batch_size = 100

epochs = 3

model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3)
prediction = model.predict(X_test)

y_pred = (prediction > 0.5).astype(int).reshape([1,-1])[0]

from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test.astype(int))))

print('Confusion matrix:')

confusion_matrix(y_pred, y_test.astype(int))