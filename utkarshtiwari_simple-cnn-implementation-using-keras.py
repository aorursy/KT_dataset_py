# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/sample_submission.csv')
x=train.drop(['label'],axis=1)

x_test=test.copy()
x=x/255

x_test=x_test/255
y=train['label']
y=pd.Categorical(y)
y=pd.get_dummies(y)
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.1,random_state=0)
x_train.shape

x_test.shape
x_val.shape
y_train.shape
y_val.shape
x_train=x_train.values

y_train=y_train.values

x_val=x_val.values

y_val=y_val.values

x_test=x_test.values
x_train=x_train.reshape(37800,28,28,1)

x_val=x_val.reshape(4200,28,28,1)

x_test=x_test.reshape(28000,28,28,1)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
model.fit(x_train, y_train,

          batch_size=128,

          epochs=15,

          verbose=1,

          validation_data=(x_val, y_val))
pred=model.predict(x_test,verbose=0)

new_pred = [np.argmax(y, axis=None, out=None) for y in pred]

output=pd.DataFrame({'ImageId':sub['ImageId'],'Label':new_pred})

output.to_csv('Digit_recognizer.csv', index=False)