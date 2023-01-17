# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/fake-news-dataset/news.csv')
data.head()
data=data.drop(["Unnamed: 0"],axis=1)
data.head(10)
import numpy as np

import random

import pprint

import pandas as pd

import tensorflow.compat.v1 as tf

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

tf.disable_eager_execution()
import json

import tensorflow as tf

import csv

import random

import numpy as np



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers





embedding_dim = 50

max_length = 54

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

training_size=3000

test_portion=.1
le = preprocessing.LabelEncoder()

le.fit(data['label'])

data['label']=le.transform(data['label'])
title=[]

text = []

labels=[]

#random.shuffle(data)

for x in range(training_size):

    title.append(data['title'][x])

    text.append(data['text'][x])

    labels.append(data['label'][x])





tokenizer1 = Tokenizer()

tokenizer1.fit_on_texts(title)

#tokenizer2 = Tokenizer()

#tokenizer2.fit_on_texts(text)

word_index1 = tokenizer1.word_index

vocab_size1=len(word_index1)

#word_index2 = tokenizer2.word_index

#vocab_size2=len(word_index2)



sequences1 = tokenizer1.texts_to_sequences(title)

padded1 = pad_sequences(sequences1,  padding=padding_type, truncating=trunc_type)

#sequences2 = tokenizer2.texts_to_sequences(text)

#padded2 = pad_sequences(sequences2,  padding=padding_type, truncating=trunc_type)



split = int(test_portion * training_size)



test_sequences1 = padded1[0:split]

training_sequences1 = padded1[split:training_size]

#test_sequences2 = padded2[0:split]

#training_sequences2 = padded2[split:training_size]

test_labels = labels[0:split]

training_labels = labels[split:training_size]
# Note this is the 100 dimension version of GloVe from Stanford

# I unzipped and hosted it on my site to make this notebook easier

embeddings_index = {};

with open('/kaggle/input/glove50d/glove.6B.50d.txt') as f:

    for line in f:

        values = line.split();

        word = values[0];

        coefs = np.asarray(values[1:], dtype='float32');

        embeddings_index[word] = coefs;



embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim));

for word, i in word_index1.items():

    embedding_vector = embeddings_index.get(word);

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector;
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size1+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(64, 5, activation='relu'),

    tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()



num_epochs = 50



training_padded = np.array(training_sequences1)

training_labels = np.array(training_labels)

testing_padded = np.array(test_sequences1)

testing_labels = np.array(test_labels)



history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)



print("Training Complete")

X="Karry to go to France in gesture of sympathy "



sequences = tokenizer1.texts_to_sequences([X])[0]

sequences = pad_sequences([sequences],maxlen=54 , padding=padding_type, truncating=trunc_type )

if(model.predict(sequences,verbose=0)[0][0] >= 0.5 ):

    print("This news is True")

else:

    print("This news is false")
