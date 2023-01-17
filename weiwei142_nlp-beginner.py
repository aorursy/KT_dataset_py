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
from keras.datasets import imdb

from keras.preprocessing import sequence
max_features = 10000

maxlen = 500



batch_size = 32
print('Reading the Data......')

# save np.load

np_load_old = np.load



# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



# call load_data with allow_pickle implicitly set to true



(input_train,y_train), (input_test,y_test) = imdb.load_data(num_words=max_features)



# restore np.load for future normal usage

np.load = np_load_old







print(len(input_train),'train sequences')

print(len(input_test),'test sequences')



print('Pad sequence(sameples x time)')

input_train = sequence.pad_sequences(input_train,maxlen=maxlen)

input_test  = sequence.pad_sequences(input_test,maxlen=maxlen)



print('input_train shape:',input_train.shape)

print('input_test shape:',input_test.shape)
from keras.layers import Dense,Embedding,SimpleRNN

from keras.models import Sequential



model = Sequential()

model.add(Embedding(max_features,32))

model.add(SimpleRNN(32))

model.add(Dense(1,activation = 'sigmoid'))



model.compile(optimizer = 'rmsprop',loss='binary_crossentropy',metrics=['acc'])



history = model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)



import matplotlib.pyplot as plt

def show_all(history):

    def show(history,acc,val_acc,label):

        epochs = range(1, len(history.history[acc])+1)

        plt.plot(epochs,history.history[acc],label='Training '+label)

        plt.plot(epochs,history.history[val_acc],label='Validation '+label)

        plt.title('Training and Validation '+label)

        plt.legend()

    plt.figure(figsize=(15,5))

    plt.subplot(121)

    show(history,'acc','val_acc','acc')

    plt.subplot(122)

    show(history,'loss','val_loss','loss')

show_all(history)
from keras.layers import LSTM



model_2 = Sequential()

model_2.add(Embedding(max_features,32))

model_2.add(LSTM(32))

model_2.add(Dense(1, activation='sigmoid'))

model_2.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history_2 = model_2.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)
show_all(history_2)