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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

train.head()
y_train = np.array(train['label'])

y_train
X_train = train.drop(['label'],1)

X_train.head()
plt.imshow(np.array(X_train.iloc[4]).reshape(28,28),cmap='gray')
from keras.utils import np_utils



y_train = np_utils.to_categorical(y_train, 10)

print(y_train[:10])
X_train = np.array(X_train/255)



# define the model

model = Sequential()

model.add(Dense(1024, input_dim = X_train.shape[1], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



# summarize the model

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 

              metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint   



# train the model

# checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 

#                                verbose=1, save_best_only=True)

# hist = model.fit(X_train, y_train, batch_size=128, epochs=10,

#           validation_split=0.2, callbacks=[checkpointer],

#           verbose=1, shuffle=True)



hist = model.fit(X_train, y_train, batch_size=128, epochs=10,

          validation_split=0.2,

          verbose=1, shuffle=True)
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
test = test.drop(['id'],1)
X_test = np.array(test/255)
prediction = model.predict(X_test)
result = []

for i in range(0,prediction.shape[0]):

    y = np.where(prediction[i] == max(prediction[i]))[0][0]

    result.append(y)

result
snn = pd.DataFrame({'label':result}).reset_index().rename(columns = {'index':'id'})

snn.head()
snn.to_csv('snn.csv',index = False)