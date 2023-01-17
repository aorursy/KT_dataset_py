# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras
import tensorflow as tf
import sys, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.isnull().any(),test.isnull().any()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
list_tokenized_train[:1]
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
batch_size = 32
epochs = 1
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.summary()
model_json = model.to_json()
with open("model_toxic_comment1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_toxic_com.h5")
print("Saved model to disk")
!ls
result = model.predict(X_te, verbose=1)
print(result)
class_col = pd.DataFrame({'toxic':result[:,0].tolist(), 'threat':result[:,3].tolist(), 'insult':result[:,4].tolist(), 'identity_hate':result[:,5].tolist()})
id_col = test["id"]
frames = [id_col, class_col]
#frames = [test, class_col]
merged = pd.concat(frames, axis=1)
print(merged)
pd.set_option('max_colwidth', 402)
batch_line = 18
merged[(batch_line-1) * 10: 10*batch_line]
merged.to_csv('toxic_comment_submit.csv', sep=',', index = False, float_format='%.1f', columns=["id","toxic","severe_toxic","obscene","threat","insult", "identity_hate" ])
!ls
