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
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.initializers import glorot_uniform

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
%matplotlib inline
dat = pd.read_csv("../input/train.tsv", sep="\t")
dat.shape, dat.head()
test_dat = pd.read_csv("../input/test.tsv", sep="\t")
test_dat.shape, test_dat.head()
reviews = {0:'negative',
    1:'somewhat negative',
    2:'neutral',
    3:'somewhat positive',
    4:'positive'}
all_dat = dat.append(test_dat)
phrases = all_dat["Phrase"].tolist()
kerasTok = text.Tokenizer()
kerasTok.fit_on_texts(phrases)
all_phrases = kerasTok.texts_to_sequences(phrases)
X = sequence.pad_sequences(all_phrases, 60)
X_train = X[:dat.shape[0], :]
X_test = X[dat.shape[0]:, :]
vocab_size = len(kerasTok.word_counts)
embed_size = 200
maxLen = 60
vocab_size, maxLen, embed_size
Y_train = np.array(dat.Sentiment)
Y_train.shape
encode = OneHotEncoder(sparse=False)
Y_train_1hot = encode.fit_transform(np.reshape(Y_train, (Y_train.shape[0], 1)))
Y_train_1hot.shape
def movie_review_analysis(input_shape, vocab_len, embed_size):
    
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = Embedding(vocab_len+1, embed_size)
    
    embeddings = embedding_layer(sentence_indices)   
    
    X = Bidirectional(LSTM(units=128, return_sequences=True))(embeddings)
    X = Dropout(rate=0.6)(X)
    
    X = Bidirectional(LSTM(units=64))(X)
    X = Dropout(rate=0.3)(X)

    X = Dense(units=5, activation='softmax')(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    return model
model = movie_review_analysis((maxLen,), vocab_size, embed_size)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
track = model.fit(X_train, Y_train_1hot, batch_size=128, epochs=3)
loss = track.history["loss"]
acc = track.history["acc"]
ep = list(range(len(loss)))
plt.plot(ep, loss)
plt.xlabel("#epochs"); plt.ylabel("loss")
plt.plot(ep, acc)
plt.xlabel("#epochs"); plt.ylabel("accuracy")
preds = model.predict(X_test, batch_size=64, verbose=1)
preds = preds.argmax(axis=1)
submit = pd.read_csv('../input/sampleSubmission.csv')
submit['Sentiment'] = preds
submit.to_csv('SubmitResult.csv', index=False)
