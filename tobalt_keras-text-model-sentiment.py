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
import keras

from keras.models import Sequential
vocablen = 10000

maxlen = 128 # for batched input

batch_size = 512
# vectorized data

# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa

# save np.load

np_load_old = np.load

# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocablen, 

                                    seed=113,

                                    start_char=1,

                                    oov_char=2,

                                    index_from=3,

                                    )



np.load = np_load_old
print(x_train[0], y_train[0])

print(x_test[0], y_test[0])

print(len(x_train), len(x_test))
# index to decode vectors into tokens and vice versa

# 0 is reserved

# word -> num

word_index = keras.datasets.imdb.get_word_index()
print(word_index['the'])

reversed_word_index = {v:k for k, v in word_index.items()}

print(reversed_word_index[1], reversed_word_index[2], reversed_word_index[3],)
# reserve special characters

word_index = {k:v+2 for k, v in word_index.items()}

word_index['<PAD>'] = 0

word_index['<START>'] = 1

word_index['<OOV>'] = 2



# limit number of words

word_index = {k:v for k, v in word_index.items() if v < vocablen}
print(word_index['<OOV>'], word_index['the'], len(word_index))
# num -> word

reversed_word_index = {v:k for k, v in word_index.items()}
print(reversed_word_index[0], reversed_word_index[1], reversed_word_index[2], reversed_word_index[3])
# pad samples

x_train = keras.preprocessing.sequence.pad_sequences(x_train, 

                            maxlen=maxlen,

                            padding='post',

                            truncating='post',

                            value=word_index['<PAD>']

                            )



x_test = keras.preprocessing.sequence.pad_sequences(x_test, 

                            maxlen=maxlen,

                            padding='post',

                            truncating='post',

                            value=word_index['<PAD>']

)
def decode(X):

    if not isinstance(X[0], list):

        # add batch dim

        X = [X]

    tokens = []

    for x in X:

        tokenized = [reversed_word_index.get(num-1, '<OOV>') for num in x]

        tokens.append(tokenized)

    return tokens
s = ' '.join(decode(x_train[0])[0])

print(s)
from keras.models import Sequential

from keras.layers import (

    Embedding,

    Bidirectional,

    Dense,

    LSTM,

    Masking,

)



model = Sequential([

    # mask before embedding

    Embedding(vocablen, 8),

    Masking(mask_value=word_index['<PAD>']),

    Masking(mask_value=word_index['<START>']),    

    Masking(mask_value=word_index['<OOV>']),

    Bidirectional(LSTM(64,

                return_sequences=True,

                )),

    Bidirectional(LSTM(32, 

                return_sequences=True,

                )),

    Bidirectional(LSTM(16)),

    Dense(8, activation='relu'),

    Dense(1, activation='sigmoid'),

])



model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 5

history = model.fit(x_train, y_train, 

      epochs=epochs, batch_size=batch_size, verbose=2,

      validation_split=0.5)
results = model.evaluate(x_test, y_test)

print(results)
model.save('keras-sentiment-classifier.h5')