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
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

false = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true['true'] = 1

false['true'] = 0
data = pd.concat([true,false])
import re

def preprocess_text(sentence):

    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower()
data['cleaned'] = data['text'].apply(preprocess_text)

data
corona_news = pd.read_csv('/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv')

corona_news['cleaned'] = corona_news['text'].apply(preprocess_text)

corona_news
list1 = ''.join(data['cleaned'].tolist()).split(' ')

list1.extend(''.join(corona_news['cleaned'].tolist()).split(' '))

unique = pd.Series(''.join(data['cleaned'].tolist()).split(' ')).unique()
item_to_index = {}

index_to_item = {}

for index,item in enumerate(unique):

    item_to_index[item] = index

    index_to_item[index] = item
def vectorize(text):

    return_list = []

    for word in text.split(' '):

        try:

            return_list.append(item_to_index[word])

        except:

            pass

    return return_list
X = data['cleaned'].apply(vectorize)
X
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(np.array(X),data['true'],test_size=0.9999)
from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling1D

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.datasets import imdb



# set parameters:

max_features = 120645

maxlen = 7916

pool_size = 4

batch_size = 32

embedding_dims = 50

filters = 250

lstm_output_size = 70

embedding_size = 128

kernel_size = 3

hidden_dims = 250

epochs = 1



print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')



print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print('Build model...')

'''

model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))

model.add(Dropout(0.25))

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

model.add(MaxPooling1D(pool_size=pool_size))

model.add(LSTM(lstm_output_size))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



'''

model = Sequential()



# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

model.add(Embedding(max_features,

                    embedding_dims,

                    input_length=maxlen))

model.add(Dropout(0.2))



# we add a Convolution1D, which will learn filters

# word group filters of size filter_length:

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

# we use max pooling:

model.add(GlobalMaxPooling1D())



# We add a vanilla hidden layer:

model.add(Dense(hidden_dims))

model.add(Dropout(0.2))

model.add(Activation('relu'))



# We project onto a single unit output layer, and squash it with a sigmoid:

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])#'''

model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test))
model.predict(X_test)
def predict(text):

    return model.predict(sequence.pad_sequences(np.array(vectorize(text)).reshape(-1,1), maxlen=maxlen))
corona_news['true'] = corona_news['cleaned'].apply(predict)
corona_news.to_csv('predictions.csv',index=False)