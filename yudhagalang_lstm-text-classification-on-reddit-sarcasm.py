# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
import pandas as pd

train_balanced_sarcasm = pd.read_csv("../input/sarcasm/train-balanced-sarcasm.csv")
train_balanced_sarcasm
df1 = train_balanced_sarcasm[['label', 'comment']].dropna()

df1
embedding_dim = 16

max_length = 100

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

training_size = int(round(df1['label'].count(), -1) * 0.8)

print(training_size)
koment = df1['comment']

labels = df1['label']

train_data = koment[0:training_size]

train_label = labels[0:training_size]

test_data = koment[training_size:]

test_label = labels[training_size:]
vocab_size = 8000

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(train_data)



word_index = tokenizer.word_index





train_seq = tokenizer.texts_to_sequences(train_data)

train_pad = pad_sequences(train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)



test_seq = tokenizer.texts_to_sequences(test_data)

test_pad = pad_sequences(test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print (vocab_size)

print(word_index)
train_seq = np.array(train_seq)

train_pad = np.array(train_pad)

test_seq = np.array(test_seq)

test_pad = np.array(test_pad)
model = Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 5

batch_size = 32



history = model.fit(train_pad, train_label, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
accr = model.evaluate(test_pad,test_label)
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(history, "accuracy")

plot_graphs(history, "loss")