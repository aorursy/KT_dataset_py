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
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)
import pandas as pd

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed
from keras.optimizers import Adam,Adadelta,Nadam,Adamax,RMSprop,SGD
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Conv2D, AveragePooling2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
def get_model(timeseries, nfeatures, nclass):
    
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(timeseries, nfeatures)))
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))

    return model
data = np.load('../input/train.npz')
X, gender, region = data['X'], data['gender'], data['region']

#X = X.reshape(20478, 128, 33, 1)
print(X.shape)

X_train, X_test, gender_train, gender_test, region_train, region_test = train_test_split(X, gender, region, test_size=0.2, random_state=2018)

publictest = np.load('../input/publictest.npz')
X_publictest, fname = publictest['X'], publictest['name']
print('train test: ', X_train.shape, X_test.shape)
print('public test: ', X_publictest.shape)
print(X.shape)
print(X.shape[2])
opt = Adam()
model = get_model(X.shape[1], X.shape[2], 3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

batch_size = 1024
nb_epochs = 1000

class_weight = {0: 2.,
                1: 2.,
                2: 1.}

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, region_train, batch_size=batch_size,class_weight=class_weight, epochs=nb_epochs, validation_data=(X_test, region_test), callbacks=callbacks_list, verbose=2)

predicts = model.predict(X_publictest, batch_size=batch_size)
predicts = np.argmax(predicts, axis=1)

region_dict = {0:'north', 1:'central', 2:'south'}
gender_dict = {0:'female', 1:'male'}
for i in range(32):
    print(fname[i], '-->', region_dict[predicts[i]])

submit = pd.DataFrame.from_dict({'id':fname, 'accent':predicts}) 
submit.to_csv('weight_adam.csv', index=False)
submit
