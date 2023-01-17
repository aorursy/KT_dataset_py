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
data = pd.read_csv('../input/data.csv')

data.columns = ['x1','x2','x3']

target = data.copy()

target.columns = ['t1','t2','t3']



data = (data.iloc[:-1,:]).iloc[10000:,:]

target = (target.iloc[1:,:]).iloc[10000:,:]
print(data.shape)

print(target.shape)

xtrain = data.values[:-10000,:]

ytrain = target.values[:-10000,:]



xtest = data.values[-10000:,:]

ytest = target.values[-10000:,:]
xtest.shape
# Define model

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense,CuDNNGRU,CuDNNLSTM,Bidirectional,Activation,Dropout,BatchNormalization

from keras.optimizers import adam,RMSprop

from keras.callbacks import ModelCheckpoint

cb = [ModelCheckpoint("model.hdf5", save_best_only=False, period=3)]



model = Sequential()

model.add(Dense(16, input_shape=(3,)))

model.add(Activation("tanh"))

model.add(Dense(3))



model.summary()



# Compile and fit model

model.compile(optimizer=adam(lr=0.001), loss="mse")



history = model.fit(xtrain,ytrain,batch_size=32, epochs=50)
history = model.fit(xtrain,ytrain,batch_size=32, epochs=50)
model.evaluate(xtest,ytest)
model.evaluate