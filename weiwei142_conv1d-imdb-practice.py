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

max_len = 500

print('Loading data...')

# save np.load

np_load_old = np.load

# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true

# restore np.load for future normal usage

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(len(x_train),'train sequence')

print(len(x_test),'test sequence')

np.load = np_load_old

print('Pad sequence (sample x time)')

x_train = sequence.pad_sequences(x_train, maxlen=max_len)

x_test = sequence.pad_sequences(x_test, maxlen=max_len)



print('x_train shape:',x_train.shape)

print('x_test shape:',x_test.shape)
from keras.models import Sequential

from keras.layers import Embedding, Conv1D, MaxPool1D, GlobalMaxPool1D, Dense

from keras.optimizers import RMSprop



model = Sequential()

model.add(Embedding(max_features,128,input_length=max_len))

model.add(Conv1D(32,7,activation='relu'))

model.add(MaxPool1D(5))

model.add(Conv1D(32,7,activation='relu'))

model.add(GlobalMaxPool1D())

model.add(Dense(1))



model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])



history = model.fit(x_train, y_train,epochs=10,batch_size=128,validation_split=0.2)
import matplotlib.pyplot as plt

def show_all(history):

    def show(history,acc,val_acc,label):

        epochs = range(1, len(history.history[acc])+1)

        plt.plot(epochs,history.history[acc],label='Training '+label)

        plt.plot(epochs,history.history[val_acc],label='Validation '+label)

        plt.legend()

        

    plt.figure(figsize=(15,5))

    

    plt.subplot(121)

    show(history,'acc','val_acc','acc')

    plt.subplot(122)

    show(history,'loss','val_loss','loss')

show_all(history)